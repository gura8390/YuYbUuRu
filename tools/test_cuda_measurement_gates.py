"""Pure mocks for CUDA benchmark gates; no executable, app, GPU or model starts."""
from pathlib import Path
import contextlib
import copy
import io
import json
import os
import tempfile
import time
import types
import unittest
from unittest import mock

import bench_common as common
import bench_cuda_route as route

ROOT = Path(__file__).resolve().parents[1]


def host_memory():
    processes = [{"pid": pid, "private_bytes": amount, "private_working_set": amount // 2,
                  "working_set": amount, "cpu_seconds": 1.0} for pid, amount in ((100, 1000), (200, 2000))]
    return {"processes": processes, "expected_process_ids": [100, 200], "exited_process_ids": [],
            "totals": {key: sum(row[key] for row in processes) for key in route.MEMORY_FIELDS},
            "cpu_seconds_total": 2.0, "elapsed_seconds": 0.01}


class HostReadTests(unittest.TestCase):
    def kernel(self, exit_code=259, open_ok=True, times_ok=True):
        def membership(handle, job, member):
            member._obj.value = True
            return True
        def exited(handle, pointer):
            pointer._obj.value = exit_code
            return True
        return types.SimpleNamespace(OpenProcess=lambda *args: 1 if open_ok else 0,
            IsProcessInJob=membership, GetExitCodeProcess=exited,
            GetProcessTimes=lambda *args: times_ok, CloseHandle=lambda *args: True)

    def owned(self, ids):
        value = object.__new__(common.OwnedTree)
        value.handle = 1
        value.ids = mock.Mock(side_effect=ids)
        return value

    def test_live_open_failure_is_not_silently_zero(self):
        with mock.patch.object(common, 'k', self.kernel(open_ok=False)):
            with self.assertRaisesRegex(OSError, 'OpenProcess failed for live owned PID'):
                self.owned([[200], [200]]).snapshot()

    def test_pid_disappearing_before_open_is_normal_exit(self):
        with mock.patch.object(common, 'k', self.kernel(open_ok=False)):
            value = self.owned([[200], []]).snapshot()
        self.assertEqual(value['exited_process_ids'], [200])
        self.assertTrue(route.host_snapshot_valid(value))

    def test_live_memory_read_failure_raises(self):
        with mock.patch.object(common, 'k', self.kernel()), mock.patch.object(common, 'p',
                types.SimpleNamespace(GetProcessMemoryInfo=lambda *args: False)):
            with self.assertRaisesRegex(OSError, 'GetProcessMemoryInfo failed for live owned PID'):
                self.owned([[200], [200]]).snapshot()

    def test_exited_memory_read_failure_is_normal_exit(self):
        with mock.patch.object(common, 'k', self.kernel(exit_code=0)), mock.patch.object(common, 'p',
                types.SimpleNamespace(GetProcessMemoryInfo=lambda *args: False)):
            value = self.owned([[200]]).snapshot()
        self.assertEqual(value['exited_process_ids'], [200])
        self.assertTrue(route.host_snapshot_valid(value))

    def test_live_cpu_time_read_failure_raises(self):
        with mock.patch.object(common, 'k', self.kernel(times_ok=False)), mock.patch.object(common, 'p',
                types.SimpleNamespace(GetProcessMemoryInfo=lambda *args: True)):
            with self.assertRaisesRegex(OSError, 'GetProcessTimes failed for live owned PID'):
                self.owned([[200], [200]]).snapshot()

    def test_sampler_records_non_oserror_instead_of_dying_silently(self):
        owned = types.SimpleNamespace()
        sampler = common.Sampler(owned, time.perf_counter(), 0)
        def fail():
            sampler.stop.set()
            raise RuntimeError('synthetic snapshot failure')
        owned.snapshot = fail
        sampler.run()
        self.assertIn('RuntimeError: synthetic snapshot failure', sampler.errors)


class BenchmarkGateTests(unittest.TestCase):
    def test_private_reports_refuse_output_outside_ignored_evidence(self):
        output = ROOT / 'report-outside-ignored-evidence.json'
        args = ['bench_cuda_route.py', '--exe', 'unused', '--app-dir', 'unused',
                '--manifest', 'unused', '--reference', 'unused', '--output', str(output)]
        with mock.patch('sys.argv', args), mock.patch.object(route.subprocess, 'Popen') as spawn, \
             contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                route.main()
        self.assertEqual(raised.exception.code, 2)
        spawn.assert_not_called()
        self.assertFalse(output.exists())

    def setUp(self):
        private = (ROOT / 'evidence/private').resolve()
        private.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='gate-mock-', dir=private)
        self.folder = Path(self.temporary.name).resolve()
        self.assertTrue(self.folder.is_relative_to(private))

    def tearDown(self):
        self.assertTrue(self.folder.is_relative_to((ROOT / 'evidence/private').resolve()))
        self.temporary.cleanup()

    def run_mock(self, *, mismatch=False, segment_error=False, returncode=0, empty_samples=False,
                 bad_gpu=False, collector_error=False, close_ok=True, host_stage_error=False):
        app = self.folder / 'app'
        app.mkdir()
        (app / 'settings.json').write_text(json.dumps({'gpu_backend': 'cuda'}), 'utf-8')
        exe = self.folder / 'not-an-executable.exe'
        exe.write_bytes(b'synthetic only; Popen is mocked')
        pcm = self.folder / 'synthetic.f32'
        pcm.write_bytes(bytes(64))
        manifest = self.folder / 'manifest.json'
        manifest.write_text(json.dumps([{'id': 'synthetic', 'pcm': str(pcm)}]), 'utf-8')
        result = {'duration': 1.0, 'text': 'synthetic', 'text_accu': 'synthetic',
                  'tokens': ['synthetic'], 'timestamps': [0.0]}
        metrics = {'text': 'synthetic', 'token_ids': [1], 'n_prefill': 2, 'n_generate': 1, 'is_aborted': False}
        ref = self.folder / 'reference.json'
        ref.write_text(json.dumps({'cleanup_confirmed': True, 'passed': True, 'mode': 'asr',
            'samples': [{'id': 'synthetic', 'pcm_sha256': route.sha(pcm), 'seed': 7,
                         'result': result, 'metrics': metrics}]}), 'utf-8')
        output = self.folder / 'report.json'
        log = output.with_suffix('.logs')
        log.mkdir()
        (log / 'rust-worker.log').write_text(
            'using device CUDA0\noffloaded 29/29 layers to GPU\nCUDA0 compute buffer size = 1 MiB\n', 'utf-8')
        class FakeOwned:
            killed = False
            def assign(self, process): pass
            def ids(self): return [] if self.killed else [100, 200]
            def snapshot(self):
                if host_stage_error:
                    raise OSError('synthetic live host read failure')
                return host_memory()
            def kill(self): self.killed = True
            def close(self): pass
        class FakeSampler:
            def __init__(self, *args):
                self.samples = [] if empty_samples else [host_memory()]
                self.errors = []
                self.thread = types.SimpleNamespace(start=lambda: None)
            def finish(self): pass
        class FakeCollector:
            def __init__(self):
                if collector_error:
                    raise RuntimeError('synthetic collector initialization failure')
            def snapshot(self, pids):
                return {'gpu_process_memory': {'sampling_ok': not bad_gpu, 'rows': [] if bad_gpu else [
                    {'pid': 200, 'adapter_luid': '0x0_0x1', 'dedicated_bytes': 1234,
                     'shared_bytes': 0, 'errors': []}]}}
            def close(self): return {'closed': close_ok}
        process = types.SimpleNamespace(pid=100, returncode=returncode,
            stdin=types.SimpleNamespace(close=lambda: None), wait=lambda timeout: None)
        def call(process, owned, timeout, request, pcm=b''):
            if request['command'] == 'segment':
                if segment_error:
                    raise RuntimeError('synthetic model/API failure')
                actual = copy.deepcopy(result)
                if mismatch:
                    actual['text'] = 'different'
                return {'result': actual, 'metrics': metrics}
            return {'kind': 'ready'}
        args = ['bench_cuda_route.py', '--exe', str(exe), '--app-dir', str(app), '--manifest', str(manifest),
                '--reference', str(ref), '--output', str(output), '--match', 'compare', '--first', '1']
        with mock.patch.object(route, 'OwnedTree', FakeOwned), mock.patch.object(route, 'Sampler', FakeSampler), \
             mock.patch.object(route, 'Collector', FakeCollector), mock.patch.object(route.subprocess, 'Popen', return_value=process), \
             mock.patch.object(route, 'receive', return_value={'kind': 'boot'}), mock.patch.object(route, 'call', side_effect=call), \
             mock.patch.object(route, 'timed_io', side_effect=lambda owned, timeout, operation: operation()), \
             mock.patch('sys.argv', args), mock.patch.dict(os.environ,
                {'GGML_CUDA_GRAPH_OPT': '1', 'GGML_CUDA_DISABLE_GRAPHS': '0'}), contextlib.redirect_stdout(io.StringIO()):
            try:
                route.main()
            except SystemExit as error:
                self.assertEqual(error.code, 1)
        return json.loads(output.read_text('utf-8'))

    def test_compare_preserves_differences_but_requires_execution(self):
        report = self.run_mock(mismatch=True)
        self.assertTrue(report['passed'])
        self.assertTrue(report['execution_passed'])
        self.assertFalse(report['outputs_exact'])

    def test_compare_does_not_accept_api_failure(self):
        report = self.run_mock(segment_error=True)
        self.assertFalse(report['passed'])
        self.assertFalse(report['execution_passed'])

    def test_compare_does_not_accept_nonzero_exit(self):
        report = self.run_mock(returncode=1)
        self.assertFalse(report['passed'])
        self.assertFalse(report['execution_passed'])

    def test_empty_sampling_is_null_and_fails_measurement(self):
        report = self.run_mock(empty_samples=True)
        self.assertTrue(report['execution_passed'])
        self.assertFalse(report['measurement_passed'])
        self.assertFalse(report['passed'])
        self.assertTrue(all(value is None for value in report['sampled_peak_totals'].values()))

    def test_bad_gpu_does_not_masquerade_as_execution_failure(self):
        report = self.run_mock(bad_gpu=True)
        self.assertTrue(report['passed'])
        self.assertTrue(report['execution_passed'])
        self.assertTrue(report['measurement_passed'])
        self.assertFalse(report['gpu_measurement_passed'])

    def test_collector_init_failure_does_not_stop_model_calls(self):
        report = self.run_mock(collector_error=True)
        self.assertTrue(report['execution_passed'])
        self.assertTrue(report['passed'])
        self.assertFalse(report['gpu_measurement_passed'])

    def test_collector_close_failure_is_gpu_measurement_failure(self):
        report = self.run_mock(close_ok=False)
        self.assertTrue(report['execution_passed'])
        self.assertTrue(report['passed'])
        self.assertFalse(report['gpu_measurement_passed'])

    def test_host_stage_failure_fails_host_measurement_not_execution(self):
        report = self.run_mock(host_stage_error=True)
        self.assertTrue(report['execution_passed'])
        self.assertFalse(report['measurement_passed'])
        self.assertFalse(report['passed'])

    def test_raw_graph_environment_zero_is_recorded_as_present(self):
        report = self.run_mock()
        self.assertEqual(report['inherited_graph_environment'],
                         {'GGML_CUDA_GRAPH_OPT': '1', 'GGML_CUDA_DISABLE_GRAPHS': '0'})
        self.assertTrue(all(report['inherited_graph_environment_present'].values()))
        self.assertTrue(report['gpu_measurement_passed'])  # Service has no GPU row.


if __name__ == '__main__':
    unittest.main()
