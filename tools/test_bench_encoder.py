"""Pure mocks: never starts a model, process, app, or real sampler thread."""
import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_encoder as bench

SHA = 'a' * 64
MANIFEST = [{'id': 'same', 'pcm': 'private.pcm', 'pcm_sha256': SHA}]


def probe():
    return {'complete': True, 'cpu_only': True, 'no_decoder_or_sampling': True,
            'settings': {'encoder_provider': 'cpu'}, 'settings_sha256': SHA,
            'manifest_sha256': SHA, 'ort_sha256': SHA, 'frontend_sha256': SHA,
            'backend_sha256': SHA, 'outputs_exact': None, 'load_and_warmup_seconds': .1,
            'samples': [{'id': 'same', 'repeat': 0, 'pcm_sha256': SHA,
                         'embedding_sha256': SHA, 'embedding_values': 4,
                         'embedding_dimension': 2, 'encode_seconds': .2,
                         'stages': {key: 0.0 for key in bench.STAGE_SECONDS}}]}


def snapshot(pid=123):
    memory = {key: 10 for key in bench.FIELDS}
    return {'processes': [{'pid': pid, 'cpu_seconds': .5, **memory}],
            'expected_process_ids': [pid], 'exited_process_ids': [],
            'totals': memory, 'cpu_seconds_total': .5}


class EncoderBenchTests(unittest.TestCase):
    def run_fake(self, data='valid', reference=None, samples=None, popen_error=None,
                 assign_error=False, timeout=False, kill_error=False, finish_error=False,
                 high_qos=False, qos_success=True):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exe = root / 'probe.exe'; exe.write_bytes(b'not executable')
            manifest = root / 'manifest.json'; manifest.write_text(json.dumps(MANIFEST))
            out = root / 'evidence' / 'result.json'
            args = ['bench', '--exe', str(exe), '--app-dir', str(root),
                    '--manifest', str(manifest), '--output', str(out)]
            if high_qos: args += ['--high-qos']
            if reference is not None:
                path = root / 'reference.json'; path.write_text(json.dumps(reference))
                args += ['--reference', str(path)]
            owned = Mock()
            owned.ids.return_value = []
            if assign_error: owned.assign.side_effect = OSError('assign failed')
            if kill_error: owned.kill.side_effect = OSError('kill failed')
            sampler = Mock()
            sampler.samples = [snapshot()] if samples is None else samples
            sampler.errors = []
            if finish_error: sampler.finish.side_effect = RuntimeError('finish failed')
            proc = Mock(pid=123, returncode=None)
            proc._handle = 456
            proc.poll.side_effect = lambda: proc.returncode
            waits = 0
            def wait(timeout=None):
                nonlocal waits
                waits += 1
                if timeout is not None and waits == 1 and timeout == 180 and timeout_failure:
                    raise subprocess.TimeoutExpired('mock', timeout)
                proc.returncode = 0
                return 0
            timeout_failure = timeout
            proc.wait.side_effect = wait
            proc.kill.side_effect = lambda: setattr(proc, 'returncode', -9)
            def launch(*args, **kwargs):
                if popen_error: raise popen_error
                if data is not None:
                    value = probe() if data == 'valid' else copy.deepcopy(data)
                    out.with_suffix('.probe.json').write_text(json.dumps(value))
                return proc
            failure = None
            def qos(process):
                self.assertIs(process, proc)
                owned.assign.assert_called_once_with(proc)
                sampler.thread.start.assert_not_called()
                return {'applied_successfully': qos_success, 'setter_passed': qos_success,
                        'setter_finished_seconds': .0001, 'errors': [] if qos_success else ['mock failure']}
            with patch.object(bench, 'ROOT', root), patch.object(bench, 'OwnedTree', return_value=owned), \
                    patch.object(bench, 'Sampler', return_value=sampler), \
                    patch.object(bench.subprocess, 'Popen', side_effect=launch) as launch_mock, \
                    patch.object(bench, 'apply_high_qos', side_effect=qos) as qos_mock, \
                    patch.object(sys, 'argv', args), contextlib.redirect_stdout(io.StringIO()):
                try: bench.main()
                except (SystemExit, ValueError) as error: failure = error
            if not high_qos: qos_mock.assert_not_called()
            result = json.loads(out.read_text()) if out.exists() else None
            return result, failure, owned, sampler, proc, launch_mock

    def test_baseline_capture_is_not_quality_acceptance(self):
        result, error, *_ = self.run_fake()
        self.assertIsNone(error)
        self.assertTrue(result['passed'])
        self.assertFalse(result['quality_checked'])
        self.assertIsNone(result['outputs_exact'])

    def test_missing_probe_fails(self):
        result, error, *_ = self.run_fake(data=None)
        self.assertFalse(result['passed']); self.assertFalse(result['execution_passed'])
        self.assertIsInstance(error, SystemExit)

    def test_bad_probe_schema_and_timings_fail(self):
        mutations = [lambda d: d.update(samples=[]), lambda d: d['samples'].append(copy.deepcopy(d['samples'][0])),
                     lambda d: d['samples'][0].pop('encode_seconds'),
                     lambda d: d['samples'][0].update(encode_seconds=float('nan')),
                     lambda d: d['samples'][0].update(encode_seconds=float('inf')),
                     lambda d: d['samples'][0].update(encode_seconds=-1),
                     lambda d: d['samples'][0]['stages'].pop('mel_seconds'),
                     lambda d: d['samples'][0].update(embedding_dimension=0),
                     lambda d: d['samples'][0].update(repeat=1), lambda d: d.update(cpu_only=False)]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                value = probe(); mutate(value)
                result, error, *_ = self.run_fake(data=value)
                self.assertFalse(result['passed']); self.assertIsInstance(error, SystemExit)

    def test_invalid_reference_prevents_launch(self):
        for key, value in [('complete', False), ('cpu_only', False), ('samples', []), ('outputs_exact', False)]:
            with self.subTest(key=key):
                ref = probe(); ref[key] = value
                _, error, _, _, _, launch = self.run_fake(reference=ref)
                self.assertIsInstance(error, ValueError); launch.assert_not_called()

    def test_reference_requires_correct_fields_and_order(self):
        for key in ('embedding_sha256', 'embedding_dimension', 'pcm_sha256', 'encode_seconds'):
            with self.subTest(key=key):
                ref = probe(); ref['samples'][0].pop(key)
                _, error, _, _, _, launch = self.run_fake(reference=ref)
                self.assertIsInstance(error, ValueError); launch.assert_not_called()

    def test_exact_reference_passes(self):
        value = probe(); value['outputs_exact'] = True
        result, error, *_ = self.run_fake(data=value, reference=probe())
        self.assertIsNone(error); self.assertTrue(result['passed']); self.assertTrue(result['outputs_exact'])

    def test_claimed_exact_does_not_hide_changed_embedding(self):
        value = probe(); value['outputs_exact'] = True; value['samples'][0]['embedding_sha256'] = 'b' * 64
        result, error, *_ = self.run_fake(data=value, reference=probe())
        self.assertFalse(result['passed']); self.assertFalse(result['outputs_exact'])

    def test_changed_model_identity_fails(self):
        value = probe(); value['outputs_exact'] = True; value['backend_sha256'] = 'b' * 64
        result, error, *_ = self.run_fake(data=value, reference=probe())
        self.assertFalse(result['passed']); self.assertIsInstance(error, SystemExit)

    def test_empty_and_no_root_measurements_fail(self):
        for samples in ([], [{'processes': [], 'expected_process_ids': [], 'totals': {key: 0 for key in bench.FIELDS}, 'cpu_seconds_total': 0}], [snapshot(999)]):
            with self.subTest(samples=samples):
                result, error, *_ = self.run_fake(samples=samples)
                self.assertFalse(result['passed']); self.assertFalse(result['measurement_passed'])
                if not samples:
                    self.assertIsNone(result['sampled_cpu_seconds_max'])
                    self.assertTrue(all(v is None for v in result['sampled_peak_totals'].values()))

    def test_popen_failure_closes_without_unstarted_join(self):
        result, error, owned, sampler, *_ = self.run_fake(popen_error=OSError('mock launch'))
        self.assertFalse(result['passed']); owned.close.assert_called_once(); sampler.finish.assert_not_called()

    def test_assign_failure_kills_unassigned_root(self):
        result, error, owned, sampler, proc, _ = self.run_fake(assign_error=True)
        self.assertFalse(result['passed']); proc.kill.assert_called_once(); owned.close.assert_called_once()
        sampler.finish.assert_not_called()

    def test_timeout_cleanup_and_receipt(self):
        result, error, owned, sampler, proc, _ = self.run_fake(timeout=True)
        self.assertFalse(result['passed']); self.assertIn('TimeoutExpired', result['error'])
        proc.kill.assert_called_once(); sampler.finish.assert_called_once(); owned.close.assert_called_once()

    def test_cleanup_failures_do_not_skip_close_or_receipt(self):
        for kwargs in ({'kill_error': True}, {'finish_error': True}):
            result, error, owned, *_ = self.run_fake(**kwargs)
            self.assertFalse(result['passed']); self.assertTrue(result['cleanup_errors']); owned.close.assert_called_once()

    def test_repeat_order_is_explicit(self):
        manifest = [MANIFEST[0], {**MANIFEST[0], 'id': 'second'}]
        expected = bench.expected_workload(manifest, 2)
        self.assertEqual([(x['id'], x['repeat']) for x in expected], [('same', 0), ('second', 0), ('same', 1), ('second', 1)])

    def test_high_qos_uses_owned_popen_after_assign_before_sampling(self):
        result, error, *_ = self.run_fake(high_qos=True)
        self.assertIsNone(error); self.assertTrue(result['passed']); self.assertTrue(result['qos_passed'])
        self.assertGreaterEqual(result['high_qos']['setter_applied_elapsed_seconds'], 0)

    def test_high_qos_failure_cannot_pass_as_applied_candidate(self):
        result, error, *_ = self.run_fake(high_qos=True, qos_success=False)
        self.assertTrue(result['execution_passed']); self.assertFalse(result['qos_passed'])
        self.assertFalse(result['passed']); self.assertIsNone(result['high_qos']['setter_applied_elapsed_seconds'])


if __name__ == '__main__':
    unittest.main()
