"""Pure mocks/sandbox data: no model, executable, GPU or process is started."""
from pathlib import Path
import copy
import json
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import bench_asr_models as bench

SHA = 'a' * 64
BINARY = 'b' * 64
CONFIG = {'asr_model': '0.6b', 'gpu_backend': 'vulkan', 'onnx_intra_threads': 8}
EXPECTED = [{'id': 'one', 'pcm_sha256': SHA, 'seed': 123}]
MANIFEST = [{'id': 'one', 'pcm_sha256': SHA, 'pcm': 'private.f32'}]


def stage(name, cpu):
    processes = [{'pid': pid, 'private_bytes': amount, 'private_working_set': amount,
                  'working_set': amount, 'cpu_seconds': cpu / 2} for pid, amount in [(10, 10), (20, 20)]]
    return {'stage': name, 'processes': processes, 'expected_process_ids': [10, 20],
            'exited_process_ids': [], 'totals': {key: 30 for key in bench.MEMORY},
            'cpu_seconds_total': cpu, 'native_gpu': {'gpu_process_memory': {
                'sampling_ok': True, 'rows': [{'pid': 20, 'adapter_luid': 'one', 'dedicated_bytes': 100,
                    'shared_bytes': 20, 'errors': []}, {'pid': 20, 'adapter_luid': 'two',
                    'dedicated_bytes': 0, 'shared_bytes': 0, 'errors': []}]}}}


def data():
    row = EXPECTED[0] | {'result': {'duration': 1, 'text': 'private transcript', 'text_accu': 'private transcript',
             'tokens': ['private token'], 'timestamps': [0]},
             'metrics': {'text': 'private transcript', 'token_ids': [42], 'n_prefill': 4, 'n_generate': 1,
                         'is_aborted': False, **{key: .1 for key in bench.STAGES}},
             'latency_seconds': 1, 'differences': [], 'raw_differences': []}
    return {'complete': True, 'passed': True, 'execution_passed': True, 'measurement_passed': True,
            'gpu_measurement_passed': True, 'cleanup_confirmed': True, 'gpu_model_offload_verified': True,
            'gpu_context_compute_verified': True, 'error': None, 'binary_sha256': BINARY,
            'settings': dict(CONFIG), 'mode': 'asr', 'samples': [row], 'outputs_exact': True,
            'total_seconds': 1, 'warmup_seconds': .5, 'stages': [stage('ready', 1), stage('after_00', 2)],
            'expected_worker_pid': 20, 'sampled_peak_totals': {key: 40 for key in bench.MEMORY}}


class ValidationTests(unittest.TestCase):
    def validate(self, report, reference=None, exact=True):
        return bench.validate_run(report, EXPECTED, reference or data(), CONFIG, BINARY, exact)

    def test_complete_same_model_run(self):
        self.assertEqual(self.validate(data()), (0, 0))

    def test_all_measurement_gates_are_mandatory(self):
        for key in ['complete', 'execution_passed', 'measurement_passed', 'gpu_measurement_passed',
                    'cleanup_confirmed', 'gpu_model_offload_verified', 'gpu_context_compute_verified']:
            with self.subTest(key=key):
                report = data(); report[key] = False
                with self.assertRaises(ValueError): self.validate(report)

    def test_cross_model_differences_observed_but_not_same_model_accepted(self):
        report = data(); report['samples'][0]['result']['text'] = 'different'
        report['samples'][0]['differences'] = ['text']; report['outputs_exact'] = False
        self.assertEqual(self.validate(report, exact=False), (1, 0))
        with self.assertRaisesRegex(ValueError, 'Same-model'): self.validate(report)

    def test_fake_exact_or_fake_difference_metadata_rejected(self):
        report = data(); report['samples'][0]['result']['text'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'Reported differences'): self.validate(report)
        report['samples'][0]['differences'] = ['text']
        with self.assertRaisesRegex(ValueError, 'False exact'): self.validate(report, exact=False)

    def test_input_seed_order_count_and_binary_identity(self):
        changes = [lambda x: x['samples'][0].update(seed=124),
                   lambda x: x['samples'][0].update(pcm_sha256='c' * 64),
                   lambda x: x.update(samples=[]), lambda x: x.update(binary_sha256='c' * 64)]
        for change in changes:
            report = data(); change(report)
            with self.assertRaises(ValueError): self.validate(report)

    def test_missing_nonfinite_and_inconsistent_timing(self):
        for change in [lambda x: x.update(total_seconds=2),
                       lambda x: x['samples'][0]['metrics'].update(t_encode=float('nan')),
                       lambda x: x['stages'][1].update(cpu_seconds_total=.1)]:
            report = data(); change(report)
            with self.assertRaises(ValueError): self.validate(report)

    def test_aborted_or_nonfinite_result_rejected_even_cross_model(self):
        for change in [lambda x: x['samples'][0]['metrics'].update(is_aborted=True),
                       lambda x: x['samples'][0]['result'].update(timestamps=[float('nan')]),
                       lambda x: x['samples'][0]['metrics'].update(n_generate=-1)]:
            report = data(); change(report)
            with self.assertRaises(ValueError): self.validate(report, exact=False)

    def test_missing_host_snapshot_or_gpu_worker_fails(self):
        report = data(); report['stages'][1]['processes'].pop()
        with self.assertRaises(ValueError): self.validate(report)
        report = data(); report['stages'][0]['native_gpu']['gpu_process_memory']['rows'] = []
        with self.assertRaises(ValueError): self.validate(report)

    def test_no_unjustified_gpu_adapter_aggregation(self):
        report = data(); report['stages'][0]['native_gpu']['gpu_process_memory']['rows'][1]['shared_bytes'] = 1
        with self.assertRaisesRegex(ValueError, 'Multiple active GPU'): self.validate(report)

    def test_settings_only_model_may_differ(self):
        large = CONFIG | {'asr_model': '1.7b'}
        bench.settings_pair(CONFIG, large)
        with self.assertRaises(ValueError): bench.settings_pair(CONFIG, large | {'onnx_intra_threads': 16})
        with self.assertRaises(ValueError): bench.settings_pair(CONFIG, CONFIG)

    def test_reference_manifest_repeated_ids_must_have_same_identity(self):
        self.assertEqual(bench.cohort(data(), MANIFEST, 'asr', 1), EXPECTED)
        self.assertEqual(bench.cohort(data(), MANIFEST * 2, 'asr', 1), EXPECTED)
        with self.assertRaises(ValueError):
            bench.cohort(data(), MANIFEST + [MANIFEST[0] | {'pcm': 'other.f32'}], 'asr', 1)
        with self.assertRaises(ValueError): bench.cohort(data(), [MANIFEST[0] | {'pcm_sha256': 'c' * 64}], 'asr', 1)

    def test_sanitized_row_has_no_text_token_or_input_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'raw.json'; path.write_text(json.dumps(data()), 'utf-8')
            row = bench.numeric_row(data(), 'small', 1, False, path, path, True)
        encoded = json.dumps(row)
        for marker in ['private transcript', 'private token', 'token_ids', '"one"', directory]:
            self.assertNotIn(marker, encoded)
        self.assertEqual(row['gpu_stage_max_dedicated_bytes'], 100)
        self.assertEqual(row['cpu_seconds'], 1)


class OwnedRunnerTests(unittest.TestCase):
    def run_mock(self, assign_error=False, timeout=False, kill_error=False):
        owned = Mock(); owned.ids.return_value = []
        if assign_error: owned.assign.side_effect = OSError('assign failed')
        if kill_error: owned.kill.side_effect = OSError('kill failed')
        proc = Mock(); proc.stdin = Mock(); proc.returncode = None
        alive = [True]; waits = [0]
        def wait(timeout=None):
            waits[0] += 1
            if timeout is not None and waits[0] == 1 and timeout == 30 and timeout_case:
                raise subprocess.TimeoutExpired('mock', timeout)
            alive[0] = False; proc.returncode = 0
        timeout_case = timeout
        proc.wait.side_effect = wait
        proc.poll.side_effect = lambda: None if alive[0] else 0
        proc.kill.side_effect = lambda: alive.__setitem__(0, False)
        with tempfile.TemporaryDirectory() as directory, patch.object(bench, 'OwnedTree', return_value=owned), \
             patch.object(bench.subprocess, 'Popen', return_value=proc) as spawn:
            receipt = bench.owned_run(['never-executed'], Path(directory) / 'raw.json', 30, {})
        return receipt, owned, proc, spawn

    def test_success_assigns_job_before_go(self):
        events = []
        with tempfile.TemporaryDirectory() as directory:
            owned = Mock(); owned.ids.return_value = []
            owned.assign.side_effect = lambda p: events.append('assign')
            proc = Mock(returncode=0); proc.poll.return_value = 0
            proc.stdin.write.side_effect = lambda b: events.append(('go', b))
            with patch.object(bench, 'OwnedTree', return_value=owned), patch.object(bench.subprocess, 'Popen', return_value=proc):
                result = bench.owned_run(['never-executed'], Path(directory) / 'raw.json', 30, {})
        self.assertEqual(events, ['assign', ('go', b'go\n')])
        self.assertTrue(result['passed']); owned.kill.assert_called_once(); owned.close.assert_called_once()

    def test_assignment_failure_never_opens_spawn_gate(self):
        receipt, owned, proc, spawn = self.run_mock(assign_error=True)
        self.assertFalse(receipt['passed']); proc.stdin.write.assert_not_called()
        proc.kill.assert_called_once(); owned.close.assert_called_once()

    def test_timeout_cleans_only_owned_job_and_bridge(self):
        receipt, owned, proc, spawn = self.run_mock(timeout=True)
        self.assertFalse(receipt['passed']); owned.kill.assert_called_once(); proc.kill.assert_called_once()
        self.assertTrue(receipt['cleanup_confirmed'])

    def test_cleanup_failure_cannot_be_success(self):
        receipt, owned, proc, spawn = self.run_mock(kill_error=True)
        self.assertFalse(receipt['passed']); self.assertTrue(receipt['cleanup_errors'])


if __name__ == '__main__':
    unittest.main()
