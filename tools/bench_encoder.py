"""Owned-process CPU encoder screening; private inputs/results stay in evidence.

No decoder, GPU session, audio upload, model conversion or global tuning is used.
Full pipeline acceptance must be measured separately after encoder screening.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import time
from bench_common import OwnedTree, Sampler
from bench_cuda_route import host_snapshot_valid
from cpu_qos import apply_high_qos

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ('private_bytes', 'private_working_set', 'working_set')
STAGE_SECONDS = ('mel_seconds', 'frontend_pack_seconds', 'frontend_run_seconds',
                 'frontend_collect_seconds', 'backend_prepare_seconds',
                 'backend_run_seconds', 'output_copy_seconds', 'total_seconds')
IDENTITY_FIELDS = ('frontend_sha256', 'backend_sha256', 'ort_sha256')
OUTPUT_FIELDS = ('embedding_sha256', 'embedding_values', 'embedding_dimension')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def positive_integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def is_sha(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def require(condition, message):
    if not condition:
        raise ValueError(message)


def expected_workload(manifest, repeats):
    require(isinstance(manifest, list) and bool(manifest), 'Manifest must contain a nonempty array')
    for row in manifest:
        require(isinstance(row, dict) and isinstance(row.get('id'), str) and bool(row['id'])
                and isinstance(row.get('pcm'), str) and bool(row['pcm'])
                and is_sha(row.get('pcm_sha256')), 'Invalid manifest row')
    return [{'id': row['id'], 'pcm_sha256': row['pcm_sha256'], 'repeat': repeat}
            for repeat in range(repeats) for row in manifest]


def validate_probe(data, expected, label='Probe'):
    require(isinstance(data, dict) and data.get('complete') is True
            and data.get('cpu_only') is True and data.get('no_decoder_or_sampling') is True,
            f'{label} must be complete and CPU encoder-only')
    require(isinstance(data.get('settings'), dict), f'{label} settings missing')
    require(data['settings'].get('encoder_provider') == 'cpu', f'{label} CPU provider missing')
    for key in (*IDENTITY_FIELDS, 'manifest_sha256', 'settings_sha256'):
        require(is_sha(data.get(key)), f'{label} invalid {key}')
    require(data.get('outputs_exact') is None or isinstance(data.get('outputs_exact'), bool),
            f'{label} invalid outputs_exact')
    require(finite_number(data.get('load_and_warmup_seconds')), f'{label} load timing missing/nonfinite')
    samples = data.get('samples')
    require(isinstance(samples, list) and bool(samples) and len(samples) == len(expected),
            f'{label} sample count mismatch or empty')
    for index, (row, wanted) in enumerate(zip(samples, expected)):
        require(isinstance(row, dict) and type(row.get('repeat')) is int
                and all(row.get(key) == value for key, value in wanted.items()),
                f'{label} input order/hash/repeat mismatch at index {index}')
        require(is_sha(row.get('embedding_sha256')) and positive_integer(row.get('embedding_values'))
                and positive_integer(row.get('embedding_dimension'))
                and row['embedding_values'] % row['embedding_dimension'] == 0,
                f'{label} output SHA/shape missing at index {index}')
        require(finite_number(row.get('encode_seconds')), f'{label} encode timing missing/nonfinite at index {index}')
        stages = row.get('stages')
        require(isinstance(stages, dict) and all(finite_number(stages.get(key)) for key in STAGE_SECONDS),
                f'{label} stage timing missing/nonfinite at index {index}')
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('exe', 'app-dir', 'manifest', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--timing', action='store_true')
    parser.add_argument('--high-qos', action='store_true', help='Diagnostic: disable EXECUTION_SPEED throttling only for this owned probe process')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--timeout-seconds', type=float, default=180)
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to((ROOT / 'evidence').resolve()):
        parser.error('Output must be under ignored evidence')
    if not 1 <= args.repeats <= 20 or not finite_number(args.timeout_seconds) or not 1 <= args.timeout_seconds <= 600:
        parser.error('Invalid repeats or timeout')
    exe = args.exe.resolve(strict=True)
    app = args.app_dir.resolve(strict=True)
    manifest = args.manifest.resolve(strict=True)
    expected = expected_workload(json.loads(manifest.read_text('utf-8')), args.repeats)
    reference = None
    if args.reference:
        reference = json.loads(args.reference.resolve(strict=True).read_text('utf-8'))
        validate_probe(reference, expected, 'Reference')
        require(reference.get('outputs_exact') is not False, 'Reference already failed its output comparison')
    out.parent.mkdir(parents=True, exist_ok=True)
    probe = out.with_suffix('.probe.json')
    stdout_path, stderr_path = out.with_suffix('.stdout.txt'), out.with_suffix('.stderr.txt')
    for path in (out, probe, stdout_path, stderr_path):
        if path.exists():
            raise FileExistsError(path)
    command = [str(exe), '--app-dir', str(app), '--manifest', str(manifest),
               '--output', str(probe), '--repeats', str(args.repeats)]
    if args.config:
        command += ['--config', str(args.config.resolve(strict=True))]
    if args.reference:
        command += ['--reference', str(args.reference.resolve(strict=True))]
    env = os.environ.copy()
    for key in ('YYUR_ENCODER_TIMING', 'YYUR_ORT_PROFILE_DIR', 'YYUR_PROFILE_LOAD'):
        env.pop(key, None)
    if args.timing:
        env['YYUR_ENCODER_TIMING'] = '1'
    report = {
        'passed': False, 'execution_passed': False, 'measurement_passed': False,
        'quality_checked': reference is not None, 'outputs_exact': None,
        'exe_sha256': sha(exe), 'reference_sha256': sha(args.reference.resolve()) if args.reference else None,
        'expected_samples': len(expected), 'timing_enabled': args.timing,
        'high_qos_requested': args.high_qos, 'high_qos': None, 'qos_passed': None,
        'scope': 'Encoder-only screening; baseline capture without a reference does not establish quality acceptance',
        'cpu_time_scope': '25ms sampled maximum of summed lifetime user+kernel CPU seconds for currently live owned PIDs; a lower bound, not final cumulative encoder CPU. Includes PCM preloading, identity/output hashing, load, warmup, encoding and teardown; exited PID work can be missed. Formal resource acceptance uses pipeline ready-to-tail CPU separately.',
        'memory_scope': '25ms observed owned-process peaks, not guaranteed lifetime peaks; includes all cohort PCM preloaded in the probe. Not deployed pipeline memory.',
        'encode_time_scope': 'Sum of wall time around encoder.encode only; excludes load/warmup, PCM preloading, identity/output hashing and JSON writing; timed diagnostics are excluded from formal resource acceptance.',
        'elapsed_scope': 'Launcher wall time including process setup, load/warmup, output hashing and cleanup; use encode_seconds_sum for encoding wall time',
        'error': None, 'cleanup_errors': [], 'no_decoder': True, 'no_audio_upload': True,
    }
    owned = proc = sampler = None
    sampler_started = False
    started = time.perf_counter()
    try:
        owned = OwnedTree()
        sampler = Sampler(owned, started, .025)
        with stdout_path.open('xb') as stdout, stderr_path.open('xb') as stderr:
            proc = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr, creationflags=0x08000000)
            report['probe_pid'] = proc.pid
            owned.assign(proc)
            if args.high_qos:
                request_started = time.perf_counter() - started
                report['high_qos'] = apply_high_qos(proc)
                report['high_qos']['request_started_elapsed_seconds'] = request_started
                report['high_qos']['request_finished_elapsed_seconds'] = time.perf_counter() - started
                setter_finished = report['high_qos'].get('setter_finished_seconds')
                report['high_qos']['setter_completed_elapsed_seconds'] = (
                    request_started + setter_finished if setter_finished is not None else None)
                report['qos_passed'] = report['high_qos'].get('applied_successfully') is True
                report['high_qos']['setter_applied_elapsed_seconds'] = (
                    report['high_qos']['setter_completed_elapsed_seconds']
                    if report['qos_passed'] and report['high_qos'].get('setter_passed') is True else None)
            sampler.thread.start()
            sampler_started = True
            proc.wait(timeout=args.timeout_seconds)
            report['exit_code'] = proc.returncode
            require(probe.is_file(), 'Probe report missing')
            data = json.loads(probe.read_text('utf-8'))
            rows = validate_probe(data, expected)
            report['probe'] = data
            report['execution_passed'] = proc.returncode == 0
            report['encode_seconds_sum'] = sum(row['encode_seconds'] for row in rows)
            report['stage_sums'] = {key: sum(row['stages'][key] for row in rows) for key in STAGE_SECONDS}
            if reference is not None:
                require(all(data[key] == reference[key] for key in IDENTITY_FIELDS), 'Model/ORT identity changed from reference')
                report['outputs_exact'] = data['outputs_exact'] is True and all(
                    all(row[key] == old[key] for key in OUTPUT_FIELDS)
                    for row, old in zip(rows, reference['samples']))
            else:
                require(data['outputs_exact'] is None, 'Unexpected output-comparison claim without a reference')
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
    finally:
        def clean(label, operation):
            try:
                return operation()
            except BaseException as error:
                report['cleanup_errors'].append(f'{label}: {type(error).__name__}: {error}')
                return None
        try:
            if owned is not None:
                clean('TerminateJobObject', owned.kill)
            if proc is not None:
                # Assignment failure leaves the root outside the Job. Use only
                # the process handle created here as the termination fallback.
                if proc.poll() is None:
                    clean('Terminate root handle', proc.kill)
                clean('Wait for root', lambda: proc.wait(timeout=15))
            if sampler_started:
                clean('Finish sampler', sampler.finish)
            report['sampling_errors'] = list(sampler.errors) if sampler is not None else []
            report['memory_samples'] = list(sampler.samples) if sampler is not None else []
            samples = report['memory_samples']
            covered = proc is not None and any(any(row.get('pid') == proc.pid for row in sample.get('processes', [])) for sample in samples)
            report['measurement_passed'] = bool(samples) and covered and not report['sampling_errors'] and all(
                host_snapshot_valid(sample) and finite_number(sample.get('cpu_seconds_total')) for sample in samples)
            report['sampled_peak_totals'] = {key: max((sample.get('totals', {}).get(key) for sample in samples
                if finite_number(sample.get('totals', {}).get(key))), default=None) for key in FIELDS}
            report['sampled_cpu_seconds_max'] = max((sample['cpu_seconds_total'] for sample in samples
                if finite_number(sample.get('cpu_seconds_total'))), default=None)
            if owned is not None:
                def wait_for_empty_job():
                    deadline = time.monotonic() + 5
                    while owned.ids() and time.monotonic() < deadline:
                        time.sleep(.02)
                    return not owned.ids()
                empty = clean('Confirm empty Job', wait_for_empty_job)
                report['cleanup_confirmed'] = empty is True and (proc is None or proc.poll() is not None)
            else:
                report['cleanup_confirmed'] = proc is None
        finally:
            if owned is not None:
                clean('Close Job handle', owned.close)
        report['elapsed_seconds'] = time.perf_counter() - started
        quality_ok = not report['quality_checked'] or report['outputs_exact'] is True
        qos_ok = not report['high_qos_requested'] or report['qos_passed'] is True
        report['passed'] = bool(report['execution_passed'] and quality_ok and report['measurement_passed']
                                and qos_ok and report.get('cleanup_confirmed') and not report['cleanup_errors'] and not report['error'])
        with out.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({key: report.get(key) for key in ('passed', 'execution_passed', 'measurement_passed',
        'quality_checked', 'outputs_exact', 'qos_passed', 'encode_seconds_sum', 'sampled_peak_totals', 'sampled_cpu_seconds_max')}))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
