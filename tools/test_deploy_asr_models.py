"""Optional-model policy and recovery checks in disposable fake app directories."""
from pathlib import Path
import ast
import contextlib
import importlib.util
import io
import json
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

import deploy_asr_models as policy
import deployment_transaction as txm

PROJECT = Path(__file__).resolve().parents[1]


class ModelOptionsPolicy(unittest.TestCase):
    def setUp(self):
        sandbox = PROJECT / 'evidence/sandbox'
        sandbox.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='model-policy-', dir=sandbox)
        self.home = Path(self.temporary.name)
        self.app, self.source = self.home / 'app', self.home / 'source'
        self.app.mkdir()
        self.source.mkdir()
        self.before = {'app_dir': '.', 'user_setting': 'preserve', 'onnx_intra_threads': 8}
        self.put(self.app, 'settings.json', json.dumps(self.before).encode())
        self.put(self.source, 'settings.json', json.dumps({**self.before, **policy.SETTINGS_PATCH}).encode())
        for name in ('start_client.exe', 'start_server.exe'):
            self.put(self.app, name, b'old-entry')
            self.put(self.source, name, b'new-entry')
        for name in policy.SHORTCUTS:
            self.put(self.source, name, b'reviewed-shortcut-' + name.encode())
        fake_models = {}
        for index, name in enumerate(policy.MODEL_FILES):
            self.put(self.source, policy.MODEL_DIR + '/' + name, bytes([index + 1]) * (index + 2))
            file = self.source / policy.MODEL_DIR / name
            fake_models[name] = (file.stat().st_size, txm.digest(file))
        self.patcher = patch.object(policy, 'MODEL_FILES', fake_models)
        self.patcher.start()
        self.put(self.app, policy.NATIVE_DIR + '/llama.dll', b'original-native')
        self.put(self.app, 'models/protected.bin', b'original-small-model')
        self.put(self.app, 'hot.txt', b'original-hotwords')
        self.protected = {key: txm.digest(self.app / key) for key in
                          (policy.NATIVE_DIR + '/llama.dll', 'models/protected.bin', 'hot.txt')}
        self.candidate = txm.digest(self.source / 'start_client.exe')
        self.plan = policy.build_plan(self.app, self.source, self.candidate, self.protected)
        self.file = self.home / 'plan.json'
        txm.atomic_json(self.file, self.plan)

    def tearDown(self):
        self.patcher.stop()
        self.temporary.cleanup()

    def put(self, base, key, value):
        file = base / key
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(value)

    def row(self, name):
        return next(r for r in self.plan['components'] if r['path'] == name)

    def tx(self):
        plan = policy.check_policy(self.plan, self.app, self.source, self.candidate)
        return txm.Transaction(plan, expected_root=self.app, source_root=self.source,
                               backup_name=policy.BACKUP_NAME, guard=lambda: None)

    def cli(self, *mode):
        with patch.object(txm, 'assert_not_running', lambda root: None), contextlib.redirect_stdout(io.StringIO()):
            return policy.main(['--app-dir', str(self.app), '--source-dir', str(self.source),
                                '--plan', str(self.file), '--candidate-sha256', self.candidate, *mode])

    def test_default_is_read_only_and_uses_new_sibling(self):
        before = txm.snapshot(self.app)
        self.cli()
        self.assertEqual(txm.snapshot(self.app), before)
        self.assertFalse((self.home / policy.BACKUP_NAME).exists())
        self.assertFalse((self.home / txm.BACKUP_NAME).exists())

    def test_only_six_components_and_original_model_shape_are_eligible(self):
        self.plan['components'].append(dict(self.row('start_server.exe'), path='native.dll', source='native.dll'))
        with self.assertRaisesRegex(ValueError, 'six'):
            self.tx()
        self.plan['components'].pop()
        self.row(policy.MODEL_DIR)['new']['files'].append({'path': 'extra.bin', 'bytes': 1, 'sha256': 'a' * 64})
        with self.assertRaisesRegex(ValueError, 'three unchanged'):
            self.tx()

    def test_candidate_and_source_mapping_are_independently_pinned(self):
        with self.assertRaisesRegex(ValueError, 'independently supplied'):
            policy.check_policy(self.plan, self.app, self.source, 'a' * 64)
        self.row('start_server.exe')['source'] = 'start_client.exe'
        with self.assertRaisesRegex(ValueError, 'Source mapping'):
            self.tx()

    def test_old_backup_cannot_be_reused(self):
        self.plan['backup'] = str(self.home / txm.BACKUP_NAME)
        with self.assertRaisesRegex(ValueError, 'independently named'):
            self.tx()

    def test_existing_unowned_model_or_shortcut_is_never_overwritten(self):
        for key in (policy.MODEL_DIR + '/owned.bin', policy.SHORTCUTS[0]):
            with self.subTest(key=key):
                self.put(self.app, key, b'user-data')
                with self.assertRaisesRegex(RuntimeError, 'Contents changed'):
                    policy.verify_current(self.plan, self.app, self.source, self.tx())
                if key.endswith('owned.bin'):
                    (self.app / key).unlink()
                    (self.app / policy.MODEL_DIR).rmdir()
                else:
                    (self.app / key).unlink()

    def test_preexisting_new_component_snapshot_is_rejected_during_plan_build(self):
        self.put(self.app, policy.SHORTCUTS[1], b'unowned shortcut')
        with self.assertRaisesRegex(ValueError, 'previously absent'):
            policy.build_plan(self.app, self.source, self.candidate, self.protected)

    def test_unknown_settings_are_retained_and_no_extra_tuning_is_patched(self):
        changed = {**self.before, **policy.SETTINGS_PATCH, 'onnx_intra_threads': 16}
        self.put(self.source, 'settings.json', json.dumps(changed).encode())
        self.row('settings.json')['new'] = txm.snapshot(self.source / 'settings.json')
        with self.assertRaisesRegex(RuntimeError, 'preserve every original key'):
            policy.verify_current(self.plan, self.app, self.source, self.tx())

    def test_native_and_other_protected_edits_are_rejected(self):
        original = self.protected[policy.NATIVE_DIR + '/llama.dll']
        self.plan['protected_sha256'].pop(policy.NATIVE_DIR + '/llama.dll')
        with self.assertRaisesRegex(ValueError, 'Every existing native file'):
            self.tx()
        self.plan['protected_sha256'][policy.NATIVE_DIR + '/llama.dll'] = original
        self.put(self.app, 'hot.txt', b'edited-hotwords')
        with self.assertRaisesRegex(RuntimeError, 'Protected file changed'):
            policy.verify_current(self.plan, self.app, self.source, self.tx())

    def test_manifest_content_tamper_is_rejected_before_execution(self):
        tx = self.tx()
        tx.prepare()
        data = json.loads(tx.manifest.read_text('utf-8'))
        data['components'][0]['new']['sha256'] = 'a' * 64
        txm.atomic_json(tx.manifest, data)
        with self.assertRaisesRegex(RuntimeError, 'reviewed content pins'):
            policy.verify_current(self.plan, self.app, self.source, self.tx())

    def test_deploy_and_installed_verify_keep_small_and_native(self):
        self.cli('--prepare-only')
        self.cli('--execute')
        self.cli('--verify-installed')
        for key, expected in self.protected.items():
            self.assertEqual(txm.digest(self.app / key), expected)
        self.assertEqual(json.loads((self.app / 'settings.json').read_text()), {**self.before, **policy.SETTINGS_PATCH})
        self.assertEqual(txm.snapshot(self.app / policy.MODEL_DIR), policy.model_snapshot())

    def load_engine(self, tx):
        spec = importlib.util.spec_from_file_location('deployment_transaction', tx.backup / 'deployment_transaction.py')
        engine = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(engine)
        return engine

    def run_recovery(self, tx):
        engine = self.load_engine(tx)
        with patch.dict(sys.modules, {'deployment_transaction': engine}), \
             patch.object(engine, 'assert_not_running', lambda root: None), \
             patch.object(sys, 'argv', ['rollback_encoder.py', '--execute']):
            runpy.run_path(str(tx.backup / 'rollback_encoder.py'), run_name='__main__')

    def test_new_sibling_recovery_is_self_contained_and_preserves_new_model(self):
        tx = self.tx()
        tx.deploy()
        wrapper = ast.parse((tx.backup / 'rollback_encoder.py').read_text('utf-8'))
        call = next(n for n in ast.walk(wrapper) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name) and n.func.id == 'backup_rollback_cli')
        self.assertEqual(ast.literal_eval(next(k.value for k in call.keywords if k.arg == 'backup_name')), policy.BACKUP_NAME)
        self.source.rename(self.home / 'source-unavailable')
        self.run_recovery(tx)
        self.assertFalse((self.app / policy.MODEL_DIR).exists())
        self.assertEqual(txm.snapshot(tx.backup / 'retired' / policy.MODEL_DIR), policy.model_snapshot())
        self.assertEqual((self.app / 'start_server.exe').read_bytes(), b'old-entry')
        self.assertEqual(json.loads((self.app / 'settings.json').read_text()), self.before)
        self.assertFalse((self.app / policy.SHORTCUTS[0]).exists())

    def test_recovery_rejects_scope_expansion_or_backup_rename_without_writes(self):
        tx = self.tx()
        tx.deploy()
        original = tx.manifest.read_bytes()
        before = txm.snapshot(self.app)
        for kind in ('expand', 'backup'):
            with self.subTest(kind=kind):
                data = json.loads(original)
                if kind == 'expand':
                    extra = {'path': 'hot.txt', 'source': None}
                    data['plan']['components'].append(extra)
                    data['components'].append({**extra, 'old': txm.snapshot(self.app / 'hot.txt'),
                                               'new': None, 'state': 'archived'})
                else:
                    data['plan']['backup'] = str(self.home / txm.BACKUP_NAME)
                txm.atomic_json(tx.manifest, data)
                with self.assertRaisesRegex(RuntimeError, 'original helper pins'):
                    self.run_recovery(tx)
                self.assertEqual(txm.snapshot(self.app), before)
        tx.manifest.write_bytes(original)

    def test_backup_name_is_one_normalized_component(self):
        plan = self.tx().plan
        for bad in ('../elsewhere', 'a/b', '/absolute', 'a\\b', 'C:escape', '.', 'NUL', 'x '):
            with self.subTest(name=bad), self.assertRaises(ValueError):
                txm.Transaction(plan, expected_root=self.app, source_root=self.source,
                                backup_name=bad, guard=lambda: None)

    def test_legacy_four_argument_recovery_api_still_uses_encoder_sibling(self):
        old_plan = self.tx().plan
        old_plan['backup'] = str(self.home / txm.BACKUP_NAME)
        tx = txm.Transaction(old_plan, expected_root=self.app, source_root=self.source,
                             guard=lambda: None)
        tx.deploy()
        with patch.object(txm, 'assert_not_running', lambda root: None), \
             patch.object(sys, 'argv', ['rollback_encoder.py', '--execute']):
            # Historical generated helpers supply these exact four arguments.
            txm.backup_rollback_cli(self.app, tx.backup, self.source, tx.plan['components'])
        self.assertEqual((self.app / 'start_server.exe').read_bytes(), b'old-entry')
        self.assertFalse((self.app / policy.MODEL_DIR).exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
