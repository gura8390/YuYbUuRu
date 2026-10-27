"""Owned-process CUDA route comparison using an explicit local reference cohort.

An output match is a regression check, not a ground-truth accuracy assessment.
Private PCM and recognized results stay in the ignored output directory.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import time

from bench_common import OwnedTree, Sampler, call, receive, timed_io, result_fields, FIELDS
from cuda_metrics import Collector

ROOT = Path(__file__).resolve().parents[1]

MEMORY_FIELDS = ('private_bytes', 'private_working_set', 'working_set')


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def host_snapshot_valid(value):
    if not isinstance(value, dict) or not isinstance(value.get('processes'), list):
        return False
    processes = value['processes']
    expected = value.get('expected_process_ids')
    if expected is None and not processes:
        return False
    if expected is not None:
        active = set(expected) - set(value.get('exited_process_ids', []))
        if {row.get('pid') for row in processes} != active:
            return False
    totals = value.get('totals')
    if not isinstance(totals, dict):
        return False
    return all(_number(totals.get(key)) and all(_number(row.get(key)) for row in processes)
               and totals[key] == sum(row[key] for row in processes) for key in MEMORY_FIELDS)


def gpu_worker_covered(value, worker_pid):
    if worker_pid is None or not isinstance(value, dict):
        return False
    gpu = value.get('gpu_process_memory', {})
    return gpu.get('sampling_ok') is True and any(
        row.get('pid') == worker_pid and not row.get('errors') and row.get('adapter_luid')
        and all(_number(row.get(key)) for key in ('dedicated_bytes', 'shared_bytes'))
        for row in gpu.get('rows', []))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', required=True, type=Path)
    parser.add_argument('--app-dir', required=True, type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--mode', choices=['asr', 'file'], default='asr')
    parser.add_argument('--match', choices=['exact', 'compare'], default='exact')
    parser.add_argument('--first', type=int, help='Optional prefix for diagnostic screening only')
    args = parser.parse_args()
    args.output = args.output.resolve()
    if not args.output.is_relative_to((ROOT / 'evidence').resolve()):
        parser.error('Reports contain private transcripts; output must stay under this checkout\'s ignored evidence directory')
    args.exe, args.app_dir = args.exe.resolve(strict=True), args.app_dir.resolve(strict=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    ref = json.loads(args.reference.read_text('utf-8'))
    # Older cross-backend comparisons deliberately have passed=false because
    # outputs differ. Validate their completed API cohort, preserving that verdict.
    assert ref.get('cleanup_confirmed') and not ref.get('error') and ref['samples']
    assert ref['mode'] == args.mode
    if args.first is not None:
        assert args.first > 0
    samples = ref['samples'][:args.first] if args.first is not None else ref['samples']
    assert len(samples) > 0
    by_id = {s['id']: s for s in json.loads(args.manifest.read_text('utf-8'))}
    for s in samples:
        assert all(k in s['result'] for k in FIELDS)
        assert sha(Path(by_id[s['id']]['pcm'])) == s['pcm_sha256']
    settings_path = args.app_dir / 'settings.json'
    settings = json.loads(settings_path.read_text('utf-8'))
    expected_backend = 'CUDA0' if settings['gpu_backend'] == 'cuda' else 'Vulkan0'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_dir = args.output.with_suffix('.logs')
    env = os.environ.copy(); env['YYUR_LOG_DIR'] = str(log_dir.resolve())
    for key in ('GGML_BACKEND_PATH', 'YYUR_PROFILE_LOAD', 'VK_INSTANCE_LAYERS', 'VK_LAYER_PATH'):
        env.pop(key, None)
    report = {'complete': False, 'execution_passed': False, 'passed': False,
              'measurement_passed': False, 'gpu_measurement_passed': False,
              'executable': str(args.exe), 'binary_sha256': sha(args.exe),
              'settings': settings, 'settings_sha256': sha(settings_path),
              'reference_sha256': sha(args.reference), 'reference_original_passed': ref.get('passed'),
              'match_policy': args.match, 'mode': args.mode, 'samples': [], 'stages': [],
              'inherited_graph_environment': {key: env.get(key) for key in
                  ('GGML_CUDA_GRAPH_OPT', 'GGML_CUDA_DISABLE_GRAPHS')},
              'inherited_graph_environment_present': {key: key in env for key in
                  ('GGML_CUDA_GRAPH_OPT', 'GGML_CUDA_DISABLE_GRAPHS')},
              'host_measurement_errors': [], 'gpu_measurement_errors': [], 'expected_worker_pid': None,
              'sampling_errors': [], 'memory_samples': [], 'sampled_peak_totals': {key: None for key in MEMORY_FIELDS},
              'measurement_gate_scope': 'measurement_passed covers host samples/stages; GPU stage evidence is independent.',
              'gpu_memory_sampling_scope': 'Phase snapshots only; transient in-request GPU peaks are not measured.',
              'quality_scope': 'Fixed output regression/comparison only; no human ground-truth WER inference.',
              'memory_scope': 'Owned service/worker private commitment and residency; WDDM GPU rows remain separate; absent rows are null.',
              'external_audio_upload': False}
    owned, metrics, proc, sampler = OwnedTree(), None, None, None
    try:
        metrics = Collector()
    except Exception as error:
        report['gpu_measurement_errors'].append(f'Collector initialization: {type(error).__name__}: {error}')
    started = time.perf_counter()

    def snapshot(name):
        stage = {'stage': name, 'elapsed_seconds': time.perf_counter()-started, 'native_gpu': None}
        try:
            stage.update(owned.snapshot())
            if not host_snapshot_valid(stage):
                raise RuntimeError('Incomplete host stage snapshot')
        except Exception as error:
            stage['host_measurement_error'] = f'{type(error).__name__}: {error}'
            report['host_measurement_errors'].append(name + ': ' + stage['host_measurement_error'])
        try:
            pids = owned.ids()
            workers = [pid for pid in pids if pid != proc.pid]
            if report['expected_worker_pid'] is None and len(workers) == 1:
                report['expected_worker_pid'] = workers[0]
            if workers != [report['expected_worker_pid']]:
                raise RuntimeError('Expected one stable model worker in the owned Job')
            if metrics is None:
                raise RuntimeError('Collector initialization failed')
            stage['native_gpu'] = metrics.snapshot(pids)
            stage['gpu_worker_covered'] = gpu_worker_covered(stage['native_gpu'], report['expected_worker_pid'])
            if not stage['gpu_worker_covered']:
                raise RuntimeError('No valid expected-worker PID/LUID dedicated/shared GPU row')
        except Exception as error:
            stage['gpu_measurement_error'] = f'{type(error).__name__}: {error}'
            report['gpu_measurement_errors'].append(name + ': ' + stage['gpu_measurement_error'])
        report['stages'].append(stage)

    try:
        with args.output.with_suffix('.stderr.txt').open('wb') as stderr:
            proc = subprocess.Popen([str(args.exe), '--app-dir', str(args.app_dir), '--config', str(settings_path),
                                     '--asr-idle-seconds', '0', 'serve'], env=env,
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                                    creationflags=0x08000000)
            owned.assign(proc); sampler = Sampler(owned, started, .025); sampler.thread.start()
            report['service_pid'] = proc.pid
            assert timed_io(owned, 30, lambda: receive(proc))['kind'] == 'boot'
            tick = time.perf_counter(); call(proc, owned, 120, {'command': 'warmup', 'id': 1})
            report['warmup_seconds'] = time.perf_counter()-tick
            log_path = log_dir/'rust-worker.log'
            log = log_path.read_text('utf-8', errors='replace')
            assert f'using device {expected_backend}' in log
            assert 'offloaded 29/29 layers to GPU' in log
            # Dynamic contexts are intentionally absent until the first prompt.
            report['gpu_model_offload_verified'] = True
            snapshot('ready')
            for i, sample in enumerate(samples):
                pcm = Path(by_id[sample['id']]['pcm']).read_bytes()
                tick = time.perf_counter()
                reply = call(proc, owned, 120, {'command': 'segment', 'id': i+2,
                    'task_id': f'route-{i}', 'source': 'file' if args.mode == 'file' else 'mic',
                    'language': 'auto', 'context': '', 'offset': 0.0, 'overlap': 4.0, 'is_final': True,
                    'time_start': time.time(), 'time_submit': time.time(), 'seed': sample['seed']}, pcm)
                duration = time.perf_counter()-tick; result = result_fields(reply)
                diff = [k for k in FIELDS if result[k] != sample['result'][k]]
                raw_diff = [k for k in ('text','token_ids','n_prefill','n_generate','is_aborted')
                            if reply['metrics'].get(k) != sample['metrics'].get(k)]
                report['samples'].append({k: sample[k] for k in ('id','pcm_sha256','seed')} |
                    {'result': result, 'metrics': reply['metrics'], 'latency_seconds': duration,
                     'differences': diff, 'raw_differences': raw_diff})
                if i == 0:
                    report['first_complete_seconds'] = time.perf_counter()-started
                snapshot(f'after_{i:02}')
            log = log_path.read_text('utf-8', errors='replace')
            assert re.search(re.escape(expected_backend) + r' compute buffer size', log)
            report['gpu_context_compute_verified'] = True
            report['complete'] = True
            report['total_seconds'] = sum(s['latency_seconds'] for s in report['samples'])
            report['median_seconds'] = statistics.median(s['latency_seconds'] for s in report['samples'])
            report['outputs_exact'] = not any(s['differences'] or s['raw_differences'] for s in report['samples'])
            report['timing_warning'] = 't_generate may reflect only the final repetition attempt; full IPC totals retain retry/context costs. Different backend token workloads need separate fixed-token replay.'
            call(proc, owned, 15, {'command': 'shutdown', 'id': 10000}); proc.stdin.close(); proc.wait(timeout=15)
            report['execution_passed'] = proc.returncode == 0
    except BaseException as error:
        report['error'] = repr(error)
    finally:
        if sampler:
            try:
                sampler.finish()
            except Exception as error:
                report['host_measurement_errors'].append(f'Sampler finish: {type(error).__name__}: {error}')
            report['sampling_errors'] = sampler.errors
            samples_valid = bool(sampler.samples) and all(host_snapshot_valid(s) for s in sampler.samples)
            report['sampled_peak_totals'] = {k:max(s['totals'][k] for s in sampler.samples) if samples_valid else None
                                            for k in MEMORY_FIELDS}
            report['memory_samples'] = sampler.samples
            if not samples_valid:
                report['host_measurement_errors'].append('Missing or incomplete host memory samples; peaks unavailable')
            report['measurement_passed'] = (samples_valid and not sampler.errors
                and not report['host_measurement_errors'] and bool(report['stages'])
                and all(host_snapshot_valid(stage) for stage in report['stages']))
        owned.kill()
        if proc is not None: proc.wait(timeout=15)
        deadline = time.monotonic()+5
        while owned.ids() and time.monotonic()<deadline: time.sleep(.02)
        report['cleanup_confirmed'] = not owned.ids(); owned.close()
        try:
            report['metrics_cleanup'] = metrics.close() if metrics is not None else {'closed': False, 'error': 'not initialized'}
        except Exception as error:
            report['metrics_cleanup'] = {'closed': False, 'error': f'{type(error).__name__}: {error}'}
        if not report['metrics_cleanup'].get('closed'):
            report['gpu_measurement_errors'].append('Collector cleanup failed')
        report['gpu_measurement_passed'] = bool(report['stages']) and not report['gpu_measurement_errors']
        report['passed'] = (report['execution_passed'] and report['measurement_passed']
                            and report['cleanup_confirmed'] and (args.match=='compare' or report.get('outputs_exact') is True))
        report['elapsed_seconds'] = time.perf_counter()-started
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2), 'utf-8')
    print(json.dumps({k:report.get(k) for k in ['passed','execution_passed','measurement_passed','gpu_measurement_passed',
                                              'outputs_exact','total_seconds','sampled_peak_totals']}))
    if not report['passed']: raise SystemExit(1)


if __name__ == '__main__':
    main()
