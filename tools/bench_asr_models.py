"""Isolated 0.6B/1.7B resource comparison; raw recognition stays in evidence.

Each process starts cold while Windows/driver caches remain intact. Cross-model
text differences are observations, never a ground-truth accuracy measurement.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time

from bench_common import FIELDS, OwnedTree
from bench_cuda_route import host_snapshot_valid, gpu_worker_covered

ROOT = Path(__file__).resolve().parents[1]
RAW_FIELDS = ('text', 'token_ids', 'n_prefill', 'n_generate', 'is_aborted')
MEMORY = ('private_bytes', 'private_working_set', 'working_set')
STAGES = ('t_encode', 't_prefill', 't_generate', 't_context')
COUNTS = {'asr': 39, 'file': 7}
MODEL = {'small': ('0.6b', 'Qwen3-ASR-0.6B', 'qwen3_asr_encoder_frontend.int4.onnx',
                   'qwen3_asr_encoder_backend.int4.onnx', 'qwen3_asr_llm.q4_k.gguf'),
         'large': ('1.7b', 'Qwen3-ASR-1.7B', 'qwen3_asr_encoder_frontend.onnx',
                   'qwen3_asr_encoder_backend.onnx', 'qwen3_asr_llm.gguf')}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def finite_tree(value):
    if type(value) in (int, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    return True


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text('utf-8'))


def save(path, data):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)


def cohort(reference, manifest, phase, count):
    require(isinstance(reference, dict) and reference.get('mode') == phase
            and reference.get('cleanup_confirmed') is True and not reference.get('error'),
            'Reference is incomplete or has the wrong phase')
    rows = reference.get('samples')
    require(isinstance(rows, list) and len(rows) == count, 'Unexpected reference cohort size')
    require(isinstance(manifest, list) and bool(manifest), 'Empty input manifest')
    by_id = {}
    for item in manifest:
        require(isinstance(item, dict) and isinstance(item.get('id'), str)
                and isinstance(item.get('pcm'), str), 'Invalid manifest row')
        if item['id'] in by_id:
            require(all(item.get(key) == by_id[item['id']].get(key) for key in ('pcm', 'pcm_sha256')),
                    'Conflicting repeated manifest ID')
        by_id[item['id']] = item
    result = []
    for row in rows:
        require(isinstance(row, dict) and row.get('id') in by_id
                and isinstance(row.get('seed'), int) and not isinstance(row.get('seed'), bool),
                'Reference input/seed is missing')
        wanted = by_id[row['id']]
        require(row.get('pcm_sha256') == wanted.get('pcm_sha256')
                and re.fullmatch(r'[a-f0-9]{64}', row.get('pcm_sha256', '')),
                'Reference input SHA differs from manifest')
        require(all(key in row.get('result', {}) for key in FIELDS)
                and all(key in row.get('metrics', {}) for key in RAW_FIELDS),
                'Reference output fields are incomplete')
        result.append({key: row[key] for key in ('id', 'pcm_sha256', 'seed')})
    return result


def settings_pair(small, large):
    require(small.get('asr_model', '0.6b') == '0.6b' and large.get('asr_model') == '1.7b',
            'App configurations must explicitly select the expected model')
    require({k: v for k, v in small.items() if k != 'asr_model'} ==
            {k: v for k, v in large.items() if k != 'asr_model'},
            'App configurations differ beyond asr_model')
    require(small.get('gpu_backend') == 'vulkan' and small.get('encoder_provider', 'cpu') == 'cpu',
            'This comparison requires the validated CPU encoder/Vulkan route')


def identities(exe, apps, configs, phase):
    result = {'binary_sha256': sha(exe), 'models': {}, 'runtime': {}}
    common = None
    for label, app in apps.items():
        spec = MODEL[label]
        require(not configs[label].get('asr_model_dir'), 'Custom model paths require a separate identity spec')
        directory = app / 'models/Qwen3-ASR' / spec[1]
        result['models'][label] = {name: sha(directory / name) for name in spec[2:]}
        dll = Path(configs[label].get('onnx_dll') or 'internal/onnxruntime/capi/onnxruntime.dll')
        native = Path(configs[label].get('llama_dir') or 'core/server/engines/llama/bin')
        dll = dll if dll.is_absolute() else app / dll
        native = native if native.is_absolute() else app / native
        paths = {dll.name: dll, **{path.name: path for path in native.glob('*.dll')}}
        require(all(name in paths for name in ('llama.dll', 'ggml.dll', 'ggml-base.dll', 'ggml-vulkan.dll')),
                'Required native runtime file missing')
        if phase == 'file':
            align = app / 'models/Qwen3-ForcedAligner/Qwen3-ForcedAligner-0.6B'
            paths.update({name: align / name for name in ('qwen3_aligner_encoder_frontend.int4.onnx',
                         'qwen3_aligner_encoder_backend.int4.onnx', 'qwen3_aligner_llm.q4_k.gguf')})
        measured = {name: sha(path) for name, path in paths.items()}
        if common is None:
            common = measured
        require(measured == common, 'The two apps have different runtime/aligner assets')
        result['runtime'][label] = measured
    return result


def owned_run(command, output, timeout, env):
    """The bridge cannot start any benchmark child until assigned to our Job."""
    bridge = ("import subprocess,sys; assert sys.stdin.buffer.readline()==b'go\\n'; "
              "sys.exit(subprocess.call(sys.argv[1:],creationflags=0x08000000))")
    receipt = {'passed': False, 'error': None, 'cleanup_errors': []}
    owned = proc = None
    started = time.perf_counter()
    try:
        owned = OwnedTree()
        with output.with_suffix('.runner.stdout.txt').open('xb') as stdout, \
             output.with_suffix('.runner.stderr.txt').open('xb') as stderr:
            proc = subprocess.Popen([sys.executable, '-c', bridge, *command], env=env,
                                    stdin=subprocess.PIPE, stdout=stdout, stderr=stderr,
                                    creationflags=0x08000000)
            owned.assign(proc)
            proc.stdin.write(b'go\n'); proc.stdin.flush(); proc.stdin.close()
            proc.wait(timeout=timeout)
            receipt['exit_code'] = proc.returncode
    except BaseException as error:
        receipt['error'] = f'{type(error).__name__}: {error}'
    finally:
        def cleanup(name, action):
            try:
                return action()
            except BaseException as error:
                receipt['cleanup_errors'].append(f'{name}: {type(error).__name__}: {error}')
        if owned:
            cleanup('terminate owned Job', owned.kill)
        if proc:
            if proc.poll() is None:
                cleanup('terminate own bridge handle', proc.kill)
            cleanup('wait bridge', lambda: proc.wait(timeout=15))
        remaining = None
        if owned:
            deadline = time.monotonic() + 5
            remaining = cleanup('query Job PIDs', owned.ids)
            while remaining and time.monotonic() < deadline:
                time.sleep(.02)
                remaining = cleanup('query Job PIDs', owned.ids)
            cleanup('close Job', owned.close)
        receipt['cleanup_confirmed'] = (remaining == [] and (proc is None or proc.poll() is not None))
        receipt['elapsed_seconds'] = time.perf_counter() - started
        receipt['passed'] = (receipt.get('exit_code') == 0 and receipt['cleanup_confirmed']
                             and not receipt['cleanup_errors'] and receipt['error'] is None)
        save(output.with_suffix('.runner.json'), receipt)
    return receipt


def gpu_memory(stage, worker):
    require(gpu_worker_covered(stage['native_gpu'], worker), 'GPU worker instance is missing')
    rows = [row for row in stage['native_gpu']['gpu_process_memory']['rows'] if row['pid'] == worker]
    require(all(not row.get('errors') and all(number(row.get(k)) for k in
                ('dedicated_bytes', 'shared_bytes')) for row in rows), 'Incomplete GPU instance')
    active = [row for row in rows if row['dedicated_bytes'] or row['shared_bytes']]
    require(len(active) <= 1, 'Multiple active GPU instances; no aggregate is inferred')
    row = (active or rows)[0]
    return {key: row[key] for key in ('dedicated_bytes', 'shared_bytes')}


def validate_run(data, expected, reference, config, binary, exact):
    require(isinstance(data, dict) and all(data.get(k) is True for k in
            ('complete', 'passed', 'execution_passed', 'measurement_passed', 'gpu_measurement_passed',
             'cleanup_confirmed', 'gpu_model_offload_verified', 'gpu_context_compute_verified'))
            and not data.get('error'), 'A required execution/measurement/GPU/cleanup gate failed')
    require(data.get('binary_sha256') == binary and data.get('settings') == config,
            'Binary or settings identity changed')
    rows = data.get('samples', [])
    require(len(rows) == len(expected), 'Run cohort size differs')
    final_changes = raw_changes = 0
    for row, wanted, old in zip(rows, expected, reference['samples']):
        require(all(row.get(key) == value for key, value in wanted.items()), 'Input order/SHA/seed differs')
        require(all(key in row.get('result', {}) for key in FIELDS)
                and all(key in row.get('metrics', {}) for key in RAW_FIELDS), 'Output fields missing')
        require(row['metrics']['is_aborted'] is False and finite_tree(row['result'])
                and finite_tree(row['metrics']), 'Aborted or nonfinite recognition output')
        require(all(type(row['metrics'][key]) is int and row['metrics'][key] >= 0
                    for key in ('n_prefill', 'n_generate')), 'Invalid token workload count')
        final = [key for key in FIELDS if row['result'][key] != old['result'][key]]
        raw = [key for key in RAW_FIELDS if row['metrics'][key] != old['metrics'][key]]
        require(row.get('differences') == final and row.get('raw_differences') == raw,
                'Reported differences do not match actual output values')
        final_changes += bool(final)
        raw_changes += bool(raw)
        require(number(row.get('latency_seconds')) and all(number(row['metrics'].get(k)) for k in STAGES),
                'Missing/nonfinite request timings')
    outputs_exact = final_changes == raw_changes == 0
    require(data.get('outputs_exact') is outputs_exact, 'False exact-output claim')
    require(not exact or outputs_exact, 'Same-model regression differs')
    require(number(data.get('total_seconds')) and math.isclose(data['total_seconds'],
            sum(row['latency_seconds'] for row in rows), rel_tol=1e-10, abs_tol=1e-9)
            and number(data.get('warmup_seconds')), 'Missing/inconsistent total or load timing')
    stages = data.get('stages', [])
    require(len(stages) == len(rows) + 1 and stages[0]['stage'] == 'ready'
            and [s['stage'] for s in stages[1:]] == [f'after_{i:02}' for i in range(len(rows))],
            'Missing phase snapshots')
    require(all(host_snapshot_valid(stage) for stage in stages), 'Invalid host snapshot')
    for stage in stages:
        require(number(stage.get('cpu_seconds_total')), 'CPU timing missing')
        gpu_memory(stage, data.get('expected_worker_pid'))
    require(stages[-1]['cpu_seconds_total'] >= stages[0]['cpu_seconds_total'], 'CPU time moved backwards')
    require(all(number(data.get('sampled_peak_totals', {}).get(k)) for k in MEMORY), 'Missing host peak')
    return final_changes, raw_changes


def numeric_row(data, label, round_id, priming, report_path, reference, own_reference):
    stages, rows = data['stages'], data['samples']
    gpu = [gpu_memory(stage, data['expected_worker_pid']) for stage in stages]
    result = {'model': MODEL[label][0], 'round': round_id, 'priming': priming,
              'raw_report': report_path.name, 'raw_report_sha256': sha(report_path),
              'reference_sha256': sha(reference), 'reference_is_same_model': own_reference,
              'requests': len(rows), 'outputs_exact': data['outputs_exact'],
              'warmup_seconds': data['warmup_seconds'], 'request_seconds': data['total_seconds'],
              'first_request_seconds': rows[0]['latency_seconds'],
              'cpu_seconds': stages[-1]['cpu_seconds_total'] - stages[0]['cpu_seconds_total']}
    result.update({'aborted_requests': sum(row['metrics']['is_aborted'] is True for row in rows),
                   'empty_text_requests': sum(not row['result']['text'].strip() for row in rows),
                   'generated_token_total': sum(row['metrics']['n_generate'] for row in rows),
                   'prefill_token_total': sum(row['metrics']['n_prefill'] for row in rows)})
    result.update({key: sum(row['metrics'][key] for row in rows) for key in STAGES})
    for key in MEMORY:
        result['peak_' + key] = data['sampled_peak_totals'][key]
        result['ready_' + key] = stages[0]['totals'][key]
        result['tail_' + key] = stages[-1]['totals'][key]
    for key in ('dedicated_bytes', 'shared_bytes'):
        result['gpu_stage_max_' + key] = max(stage[key] for stage in gpu)
        result['gpu_tail_' + key] = gpu[-1][key]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('exe', 'small-app', 'large-app', 'manifest', 'small-reference', 'output-dir'):
        parser.add_argument('--' + key, required=True, type=Path)
    parser.add_argument('--file-manifest', type=Path)
    parser.add_argument('--file-reference', type=Path)
    parser.add_argument('--phase', choices=COUNTS, default='asr')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--timeout-seconds', type=int, default=600)
    args = parser.parse_args()
    require(1 <= args.rounds <= 9 and 30 <= args.timeout_seconds <= 1800, 'Invalid rounds/timeout')
    out = args.output_dir.resolve()
    require(out.is_relative_to((ROOT / 'evidence/large-model').resolve()), 'Output must stay under evidence/large-model')
    require(not out.exists(), 'Refusing an existing output directory')
    if args.phase == 'file':
        require(args.file_manifest and args.file_reference, 'File phase needs --file-manifest and --file-reference')
    manifest = (args.file_manifest if args.phase == 'file' else args.manifest).resolve(strict=True)
    small_ref = (args.file_reference if args.phase == 'file' else args.small_reference).resolve(strict=True)
    ref = read(small_ref)
    inputs = read(manifest)
    expected = cohort(ref, inputs, args.phase, COUNTS[args.phase])
    by_id = {row['id']: row for row in inputs}
    for row in expected:
        require(sha(Path(by_id[row['id']]['pcm'])) == row['pcm_sha256'], 'Local PCM identity mismatch')
    exe = args.exe.resolve(strict=True)
    apps = {'small': args.small_app.resolve(strict=True), 'large': args.large_app.resolve(strict=True)}
    configs = {label: read(app / 'settings.json') for label, app in apps.items()}
    settings_pair(configs['small'], configs['large'])
    assets = identities(exe, apps, configs, args.phase)
    summary = {'passed': False, 'phase': args.phase, 'rounds_per_model': args.rounds, 'identity': assets,
               'manifest_sha256': sha(manifest), 'small_reference_sha256': sha(small_ref), 'runs': [],
               'quality_scope': 'No ground-truth CER/WER. Cross-model differences are counted, not quality verdicts; each model must match its own fixed-seed reference.',
               'timing_scope': 'Fresh process/model load with Windows/driver caches retained. Warmup and complete IPC are separate; t_generate may cover only the final retry.',
               'memory_scope': '25ms observed owned-tree host peaks; GPU phase maxima are separate and are not transient peaks. One active adapter instance required.',
               'error_type': None}
    env = {key: value for key, value in os.environ.items() if not key.startswith(('YYUR_', 'GGML_'))
           and key not in ('VK_INSTANCE_LAYERS', 'VK_LAYER_PATH')}
    out.mkdir(parents=True)
    private = {'commands': [], 'error': None}
    references = {'small': small_ref, 'large': small_ref}
    order = [('small', 0), ('large', 0)]
    for index in range(1, args.rounds + 1):
        order.extend((label, index) for label in (('small', 'large') if index % 2 else ('large', 'small')))
    try:
        for label, round_id in order:
            priming = round_id == 0
            report = out / f'{label}-{"priming" if priming else f"r{round_id}"}.json'
            compare = label == 'large' and priming
            reference = references[label]
            command = [sys.executable, str(ROOT / 'tools/bench_cuda_route.py'), '--exe', str(exe),
                       '--app-dir', str(apps[label]), '--manifest', str(manifest), '--reference', str(reference),
                       '--output', str(report), '--mode', args.phase, '--match', 'compare' if compare else 'exact']
            private['commands'].append(command)
            receipt = owned_run(command, report, args.timeout_seconds, env)
            require(receipt['passed'] and report.is_file(), 'Owned child benchmark failed')
            data = read(report)
            require(data.get('mode') == args.phase, 'Child phase changed')
            require(data.get('reference_sha256') == sha(reference), 'Reference changed during a child run')
            differences = validate_run(data, expected, read(reference), configs[label], assets['binary_sha256'], not compare)
            worker_log = report.with_suffix('.logs') / 'rust-worker.log'
            log = worker_log.read_text('utf-8', errors='replace').replace('\\', '/').lower()
            spec = MODEL[label]
            require((spec[1] + '/' + spec[4]).lower() in log, 'Actual native ASR model not established by log')
            dimension = re.search(r'\bn_embd\s*=\s*(\d+)\b', log)
            require(dimension is not None and int(dimension[1]) == (1024 if label == 'small' else 2048),
                    'Actual ASR embedding dimension differs from the selected model')
            row = numeric_row(data, label, round_id, priming, report, reference, not compare)
            row.update({'final_different_requests': differences[0], 'raw_different_requests': differences[1],
                        'worker_log_sha256': sha(worker_log)})
            summary['runs'].append(row)
            if compare:
                references['large'] = report
            print(json.dumps({'phase': args.phase, 'model': spec[0], 'round': round_id,
                              'priming': priming, 'passed': True, 'request_seconds': row['request_seconds']}), flush=True)
        require(identities(exe, apps, configs, args.phase) == assets, 'Assets changed during the experiment')
        require(sha(manifest) == summary['manifest_sha256'] and sha(small_ref) == summary['small_reference_sha256'],
                'Manifest or accepted small-model reference changed')
        for row in expected:
            require(sha(Path(by_id[row['id']]['pcm'])) == row['pcm_sha256'], 'Local PCM changed during the experiment')
        require(all(read(app / 'settings.json') == configs[label] for label, app in apps.items()),
                'Settings changed during the experiment')
        summary['medians'] = {}
        for label in apps:
            rows = [r for r in summary['runs'] if r['model'] == MODEL[label][0] and not r['priming']]
            require(len(rows) == args.rounds, 'Independent process round missing')
            summary['medians'][MODEL[label][0]] = {key: statistics.median(row[key] for row in rows)
                 for key in rows[0] if type(rows[0][key]) in (int, float) and key != 'round'}
        summary['passed'] = True
    except BaseException as error:
        summary['error_type'] = type(error).__name__
        private['error'] = f'{type(error).__name__}: {error}'
    finally:
        save(out / 'orchestration-private.json', private)
        save(out / 'summary.json', summary)
        if summary['runs']:
            with (out / 'summary.csv').open('x', encoding='utf-8-sig', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(summary['runs'][0]))
                writer.writeheader(); writer.writerows(summary['runs'])
    print(json.dumps({'passed': summary['passed'], 'phase': args.phase, 'completed_runs': len(summary['runs']),
                      'error_type': summary['error_type']}))
    if not summary['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
