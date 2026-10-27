"""Encoder policy checks in temporary fake apps; no real application is touched."""
from pathlib import Path
import contextlib
import io
import ast
import importlib.util
import json
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

import deploy_encoder as policy
import deployment_transaction as txm

PROJECT = Path(__file__).resolve().parents[1]


class Policy(unittest.TestCase):
    def setUp(self):
        sandbox = PROJECT / 'evidence/sandbox'
        sandbox.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='encoder-policy-', dir=sandbox)
        self.home = Path(self.temporary.name)
        self.app, self.source = self.home / 'app', self.home / 'source'
        self.app.mkdir()
        self.source.mkdir()
        self.before = {'app_dir': '.', 'user_setting': 'preserve', 'gpu_backend': 'vulkan'}
        self.plan = {'schema': 1, 'status': 'ready', 'root': str(self.app),
                     'source_root': str(self.source), 'backup': str(self.home / txm.BACKUP_NAME),
                     'settings_before': self.before, 'settings_patch': policy.SETTINGS_PATCH,
                     'components': [], 'protected_sha256': {}}
        for key in policy.COMPONENTS:
            old, new = b'old', b'new-candidate'
            if key == 'settings.json':
                old = json.dumps(self.before).encode()
                new = json.dumps({**self.before, **policy.SETTINGS_PATCH}).encode()
            elif key.endswith('llama.dll'):
                new = b'new-llama'
            self.put(self.app, key, old)
            self.put(self.source, key, new)
            self.plan['components'].append({'path': key, 'source': key,
                                           'old_sha256': txm.digest(self.app / key),
                                           'new_sha256': txm.digest(self.source / key)})
        self.put(self.app, 'models/protected.bin', b'unchanged')
        self.plan['protected_sha256']['models/protected.bin'] = txm.digest(self.app / 'models/protected.bin')
        self.file = self.home / 'plan.json'
        txm.atomic_json(self.file, self.plan)

    def tearDown(self):
        self.temporary.cleanup()

    def put(self, base, key, value):
        file = base / key
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(value)

    def tx(self):
        p = policy.check_policy(self.plan, self.app, self.source)
        return txm.Transaction(p, expected_root=self.app, source_root=self.source, guard=lambda: None)

    def test_default_check_is_read_only(self):
        original = txm.snapshot(self.app)
        with patch.object(txm, 'assert_not_running', lambda root: None), contextlib.redirect_stdout(io.StringIO()):
            policy.main(['--app-dir', str(self.app), '--source-dir', str(self.source), '--plan', str(self.file)])
        self.assertEqual(txm.snapshot(self.app), original)
        self.assertFalse((self.home / txm.BACKUP_NAME).exists())

    def test_draft_with_missing_final_hash_is_rejected(self):
        self.plan['status'] = 'draft'
        self.plan['components'][0]['new_sha256'] = None
        with self.assertRaisesRegex(ValueError, 'finalized'):
            self.tx()

    def test_extra_file_or_different_backup_is_rejected(self):
        extra = dict(self.plan['components'][0], path='core/other.dll', source='core/other.dll')
        self.plan['components'].append(extra)
        with self.assertRaisesRegex(ValueError, 'exactly'):
            self.tx()
        self.plan['components'].pop()
        self.plan['backup'] = str(self.home / 'old-prefill-backup')
        with self.assertRaisesRegex(ValueError, 'separately named'):
            self.tx()

    def test_source_settings_must_preserve_unknown_keys(self):
        changed = dict(policy.SETTINGS_PATCH)
        self.put(self.source, 'settings.json', json.dumps(changed).encode())
        next(row for row in self.plan['components'] if row['path'] == 'settings.json')['new_sha256'] = txm.digest(self.source / 'settings.json')
        with self.assertRaisesRegex(RuntimeError, 'preserve every'):
            policy.verify_current(self.plan, self.app, self.source, self.tx())

    def test_protected_model_edit_is_rejected(self):
        self.put(self.app, 'models/protected.bin', b'user edit')
        with self.assertRaisesRegex(RuntimeError, 'Hash mismatch'):
            policy.verify_current(self.plan, self.app, self.source, self.tx())

    def test_launcher_source_edit_is_rejected(self):
        self.put(self.source, 'start_server.exe', b'not reviewed')
        with self.assertRaisesRegex(RuntimeError, 'Hash mismatch'):
            policy.verify_current(self.plan, self.app, self.source, self.tx())

    def test_manifest_must_retain_reviewed_pins(self):
        tx = self.tx()
        tx.prepare()
        self.plan['components'][0]['old_sha256'] = 'f' * 64
        with self.assertRaisesRegex(RuntimeError, 'approved hash pins'):
            policy.check_manifest_pins(tx, self.plan)

    def test_finalize_requires_explicit_matching_candidate_hash(self):
        draft = dict(self.plan, status='draft', source_root=None)
        with self.assertRaisesRegex(RuntimeError, 'explicitly supplied'):
            policy.finalized_plan(draft, self.app, self.source, 'f' * 64)
        final = policy.finalized_plan(draft, self.app, self.source, txm.digest(self.source / 'start_client.exe'))
        self.assertEqual(final['status'], 'ready')
        self.assertEqual(draft['status'], 'draft')

    def test_execute_and_installed_verify_use_only_allowed_files(self):
        with patch.object(txm, 'assert_not_running', lambda root: None), contextlib.redirect_stdout(io.StringIO()):
            args = ['--app-dir', str(self.app), '--source-dir', str(self.source), '--plan', str(self.file)]
            policy.main([*args, '--execute'])
            policy.main([*args, '--verify-installed'])
        self.assertTrue((self.home / txm.BACKUP_NAME / 'rollback_encoder.py').is_file())
        self.assertEqual((self.app / 'models/protected.bin').read_bytes(), b'unchanged')
        for row in self.plan['components']:
            self.assertEqual(txm.digest(self.app / row['path']), row['new_sha256'])

    def test_encoder_helper_pins_exact_four_files_and_rejects_extension(self):
        tx = self.tx()
        tx.deploy()
        helper = tx.backup / 'rollback_encoder.py'
        tree = ast.parse(helper.read_text('utf-8'))
        call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name) and n.func.id == 'backup_rollback_cli')
        pins = ast.literal_eval(call.args[3])
        self.assertEqual(len(pins), 4)
        self.assertEqual({row['path'] for row in pins}, set(policy.COMPONENTS))
        data = json.loads(tx.manifest.read_text('utf-8'))
        extra = {'path': 'models/protected.bin', 'source': None}
        data['plan']['components'].append(extra)
        data['components'].append({**extra, 'old': None, 'new': txm.snapshot(self.app / extra['path']), 'state': 'installed'})
        txm.atomic_json(tx.manifest, data)
        before = txm.snapshot(self.app)
        spec = importlib.util.spec_from_file_location('deployment_transaction', tx.backup / 'deployment_transaction.py')
        engine = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(engine)
        with patch.dict(sys.modules, {'deployment_transaction': engine}), \
             patch.object(engine, 'assert_not_running', lambda root: None), \
             patch.object(sys, 'argv', ['rollback_encoder.py', '--execute']), \
             self.assertRaisesRegex(RuntimeError, 'original helper pins'):
            runpy.run_path(str(helper), run_name='__main__')
        self.assertEqual(txm.snapshot(self.app), before)


if __name__ == '__main__':
    unittest.main(verbosity=2)
