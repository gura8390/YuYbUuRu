"""Destructive operations are confined to new TemporaryDirectory sandboxes."""
from pathlib import Path
import importlib.util
import ast
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
import runpy
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
(PROJECT / 'evidence/sandbox').mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location('encoder_transaction', PROJECT / 'tools/deployment_transaction.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class Interrupted(BaseException):
    pass


class Sandbox(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='transaction-', dir=PROJECT / 'evidence/sandbox')
        self.home = Path(self.temporary.name)
        assert self.home.resolve().is_relative_to((PROJECT / 'evidence/sandbox').resolve())
        self.root = self.home / 'original-app'
        self.source = self.home / 'source-app'
        self.root.mkdir()
        self.source.mkdir()
        self.put(self.root, 'core/main.py', b'old-core')
        self.put(self.root, 'models/original-model.bin', b'original-model-weights')
        self.put(self.root, 'internal/python313.dll', b'old-runtime')
        self.put(self.root, 'config_client.py', b'old-config')
        self.put(self.root, 'LLM/role.py', b'private-secret-not-for-logs')
        self.put(self.root, 'hot.txt', b'protected-hotwords')
        self.put(self.root, '2026/recording.wav', b'protected-user-recording')
        self.put(self.source, 'core/main.py', b'new-core')
        self.put(self.source, 'models/model.bin', b'new-model')
        self.put(self.source, 'internal/python313.dll', b'new-runtime')
        (self.source / 'internal/_tcl_data').mkdir()
        (self.source / 'internal/_tk_data').mkdir()
        self.put(self.source, 'config_client.py', b'new-config')
        self.put(self.source, 'README.txt', b'new-entry')
        self.plan = {'root': str(self.root), 'backup': str(self.home / mod.BACKUP_NAME),
                     'components': [{'path': key, 'source': key} for key in
                                    ('core', 'models', 'internal', 'config_client.py', 'README.txt')]
                                   + [{'path': 'LLM', 'source': None}, {'path': 'missing', 'source': None}]}
        self.original = mod.snapshot(self.root)

    def tearDown(self):
        assert self.home.resolve().is_relative_to((PROJECT / 'evidence/sandbox').resolve())
        self.temporary.cleanup()

    def put(self, base, key, content):
        file = base / key
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(content)

    def tx(self, fault=None, guard=None):
        fired = False
        def checkpoint(event, row=None):
            nonlocal fired
            if event == fault and not fired:
                fired = True
                raise Interrupted(event)
        return mod.Transaction(self.plan, expected_root=self.root, source_root=self.source,
                               guard=guard or (lambda: None), checkpoint=checkpoint)

    def assert_original(self):
        self.assertEqual(mod.snapshot(self.root), self.original)

    def assert_deployed(self):
        for row in self.plan['components']:
            if row['source'] is not None:
                self.assertEqual(mod.snapshot(self.root / row['path']), mod.snapshot(self.source / row['source']))
            else:
                self.assertFalse((self.root / row['path']).exists())
        self.assertEqual((self.root / 'hot.txt').read_bytes(), b'protected-hotwords')
        self.assertEqual((self.root / '2026/recording.wav').read_bytes(), b'protected-user-recording')

    def test_check_is_read_only(self):
        tx = self.tx()
        self.assertFalse(tx.check()['mutated'])
        self.assertFalse(tx.backup.exists())
        self.assert_original()

    def test_prepare_keeps_original_and_empty_directories(self):
        tx = self.tx()
        tx.prepare()
        self.assert_original()
        for name in ('_tcl_data', '_tk_data'):
            self.assertTrue((tx.backup / 'prepared/internal' / name).is_dir())
        self.assertTrue((tx.backup / 'rollback_encoder.py').is_file())
        self.assertTrue((tx.backup / 'deployment_transaction.py').is_file())
        self.assertNotIn('private-secret-not-for-logs', tx.manifest.read_text('utf-8'))

    def test_deploy_rollback_preserves_both_versions(self):
        tx = self.tx()
        tx.deploy()
        self.assert_deployed()
        self.assertEqual((tx.backup / 'original/LLM/role.py').read_bytes(), b'private-secret-not-for-logs')
        tx.rollback()
        self.assert_original()
        self.assertEqual((tx.backup / 'retired/core/main.py').read_bytes(), b'new-core')
        self.assertEqual((tx.backup / 'retired/models/model.bin').read_bytes(), b'new-model')
        self.assertEqual(json.loads(tx.manifest.read_text('utf-8'))['phase'], 'rolled_back')

    def test_nested_cuda_addition_keeps_vulkan_and_preserves_retired_cuda(self):
        self.put(self.root, 'core/server/engines/llama/bin/ggml-vulkan.dll', b'original-vulkan')
        self.put(self.root, 'settings.json', b'original-settings')
        self.put(self.source, 'core/server/engines/llama/cuda/bin/ggml-cuda.dll', b'isolated-cuda')
        self.put(self.source, 'settings.json', b'new-settings')
        self.plan['components'] = [
            {'path': 'core/server/engines/llama/cuda/bin', 'source': 'core/server/engines/llama/cuda/bin'},
            {'path': 'settings.json', 'source': 'settings.json'},
        ]
        tx = self.tx()
        tx.deploy()
        self.assertEqual((self.root / 'core/server/engines/llama/bin/ggml-vulkan.dll').read_bytes(), b'original-vulkan')
        self.assertEqual((self.root / 'core/server/engines/llama/cuda/bin/ggml-cuda.dll').read_bytes(), b'isolated-cuda')
        tx.rollback()
        self.assertEqual((self.root / 'settings.json').read_bytes(), b'original-settings')
        self.assertEqual((self.root / 'core/server/engines/llama/bin/ggml-vulkan.dll').read_bytes(), b'original-vulkan')
        self.assertFalse((self.root / 'core/server/engines/llama/cuda/bin').exists())
        self.assertEqual((tx.backup / 'retired/core/server/engines/llama/cuda/bin/ggml-cuda.dll').read_bytes(), b'isolated-cuda')

    def test_deploy_and_rollback_are_idempotent(self):
        tx = self.tx()
        tx.deploy()
        self.tx().deploy()
        self.assert_deployed()
        self.tx().rollback()
        self.tx().rollback()
        self.assert_original()

    def test_backup_recovery_entry_works_without_workspace_lookup(self):
        tx = self.tx()
        tx.deploy()
        # Recovery must remain self-contained even if the staged source is unavailable.
        unavailable = self.home / 'source-unavailable'
        assert self.source.resolve().is_relative_to(self.home.resolve())
        assert unavailable.parent.resolve() == self.home.resolve()
        self.source.rename(unavailable)
        self.assertFalse(self.source.exists())
        spec = importlib.util.spec_from_file_location('deployment_transaction', tx.backup / 'deployment_transaction.py')
        engine = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(engine)
        with patch.dict(sys.modules, {'deployment_transaction': engine}), \
             patch.object(engine, 'assert_not_running', lambda root: None), \
             patch.object(sys, 'argv', ['rollback_encoder.py', '--execute']):
            runpy.run_path(str(tx.backup / 'rollback_encoder.py'), run_name='__main__')
        self.assert_original()

    def run_recovery(self, tx):
        spec = importlib.util.spec_from_file_location('deployment_transaction', tx.backup / 'deployment_transaction.py')
        engine = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(engine)
        with patch.dict(sys.modules, {'deployment_transaction': engine}), \
             patch.object(engine, 'assert_not_running', lambda root: None), \
             patch.object(sys, 'argv', ['rollback_encoder.py', '--execute']):
            runpy.run_path(str(tx.backup / 'rollback_encoder.py'), run_name='__main__')

    def test_recovery_rejects_consistent_scope_extension_without_target_writes(self):
        tx = self.tx()
        tx.deploy()
        data = json.loads(tx.manifest.read_text('utf-8'))
        extra = {'path': 'hot.txt', 'source': 'hot.txt'}
        data['plan']['components'].append(extra)
        data['components'].append({**extra, 'old': None, 'new': mod.snapshot(self.root / 'hot.txt'), 'state': 'installed'})
        mod.atomic_json(tx.manifest, data)
        before, manifest = mod.snapshot(self.root), tx.manifest.read_bytes()
        with self.assertRaisesRegex(RuntimeError, 'original helper pins'):
            self.run_recovery(tx)
        self.assertEqual(mod.snapshot(self.root), before)
        self.assertEqual(tx.manifest.read_bytes(), manifest)

    def test_recovery_rejects_changed_root_without_target_writes(self):
        tx = self.tx()
        tx.deploy()
        other = self.home / 'other-app'
        self.put(other, 'valuable.txt', b'never-overwrite')
        data = json.loads(tx.manifest.read_text('utf-8'))
        data['plan']['root'] = str(other)
        mod.atomic_json(tx.manifest, data)
        before, other_before = mod.snapshot(self.root), mod.snapshot(other)
        manifest = tx.manifest.read_bytes()
        with self.assertRaisesRegex(RuntimeError, 'original helper pins'):
            self.run_recovery(tx)
        self.assertEqual(mod.snapshot(self.root), before)
        self.assertEqual(mod.snapshot(other), other_before)
        self.assertEqual(tx.manifest.read_bytes(), manifest)

    def test_recovery_rejects_duplicate_scope_with_same_count(self):
        tx = self.tx()
        tx.deploy()
        data = json.loads(tx.manifest.read_text('utf-8'))
        data['plan']['components'][-1] = dict(data['plan']['components'][0])
        data['components'][-1] = dict(data['components'][0])
        mod.atomic_json(tx.manifest, data)
        before = mod.snapshot(self.root)
        with self.assertRaisesRegex(RuntimeError, 'Duplicate recovery component'):
            self.run_recovery(tx)
        self.assertEqual(mod.snapshot(self.root), before)

    def test_recovery_rejects_manifest_component_mapping_change(self):
        tx = self.tx()
        tx.deploy()
        data = json.loads(tx.manifest.read_text('utf-8'))
        data['components'][0]['source'] = 'different-source'
        mod.atomic_json(tx.manifest, data)
        before = mod.snapshot(self.root)
        with self.assertRaisesRegex(RuntimeError, 'original helper pins'):
            self.run_recovery(tx)
        self.assertEqual(mod.snapshot(self.root), before)

    def ancestor_link(self):
        outside = self.home / 'physical'
        outside.mkdir()
        self.put(outside, 'parent/app/valuable.txt', b'never-follow-or-modify')
        link = self.home / 'ancestor-link'
        if os.name == 'nt':
            proc = subprocess.run(['cmd.exe', '/c', 'mklink', '/J', str(link), str(outside)], capture_output=True)
            if proc.returncode:
                self.skipTest('Junction creation unavailable in sandbox')
        else:
            link.symlink_to(outside, target_is_directory=True)
        return outside, link

    def test_contained_rejects_junction_above_base(self):
        outside, link = self.ancestor_link()
        try:
            base = link / 'parent/app'
            with self.assertRaisesRegex(RuntimeError, 'Links/reparse'):
                mod.contained(base, base / 'valuable.txt')
            self.assertEqual((outside / 'parent/app/valuable.txt').read_bytes(), b'never-follow-or-modify')
        finally:
            if os.name == 'nt': os.rmdir(link)
            else: link.unlink()

    def test_transaction_boundaries_reject_ancestor_above_base_parent(self):
        outside, link = self.ancestor_link()
        try:
            app = link / 'parent/app'
            plan = {'root': str(app), 'backup': str(app.parent / mod.BACKUP_NAME),
                    'components': [{'path': 'valuable.txt', 'source': None}]}
            with self.assertRaisesRegex(RuntimeError, 'Links/reparse'):
                mod.Transaction(plan, expected_root=app, source_root=self.source, guard=lambda: None)
            self.assertEqual((outside / 'parent/app/valuable.txt').read_bytes(), b'never-follow-or-modify')
            self.assertFalse((outside / 'parent' / mod.BACKUP_NAME).exists())
        finally:
            if os.name == 'nt': os.rmdir(link)
            else: link.unlink()

    def test_exclusive_transaction_lock(self):
        tx = self.tx()
        with mod.transaction_lock(tx.backup):
            with self.assertRaises((RuntimeError, OSError)):
                with mod.transaction_lock(tx.backup):
                    self.fail('Concurrent transaction lock was accepted')

    def test_read_only_source_file_is_supported(self):
        import stat
        file = self.source / 'core/main.py'
        file.chmod(stat.S_IREAD)
        try:
            self.tx().deploy()
            self.assert_deployed()
            self.tx().rollback()
            self.assert_original()
        finally:
            # Clear read-only flags only inside this temporary sandbox for cleanup.
            for base in (self.root, self.source, self.home / mod.BACKUP_NAME):
                if base.exists():
                    for candidate in base.rglob('*'):
                        if candidate.is_file(): candidate.chmod(stat.S_IREAD | stat.S_IWRITE)

    def test_stage_interruption_can_resume(self):
        with self.assertRaises(Interrupted):
            self.tx('after_stage_file').deploy()
        self.assert_original()
        self.tx().deploy()
        self.assert_deployed()

    def test_stage_interruption_can_rollback(self):
        with self.assertRaises(Interrupted):
            self.tx('after_stage_file').deploy()
        self.tx().rollback()
        self.assert_original()

    def test_resume_after_archiving_original_before_manifest(self):
        with self.assertRaises(Interrupted):
            self.tx('after_archive_rename').deploy()
        self.tx().deploy()
        self.assert_deployed()

    def test_rollback_after_archiving_original_before_manifest(self):
        with self.assertRaises(Interrupted):
            self.tx('after_archive_rename').deploy()
        self.tx().rollback()
        self.assert_original()

    def test_resume_after_install_before_manifest(self):
        with self.assertRaises(Interrupted):
            self.tx('after_install_rename').deploy()
        self.tx().deploy()
        self.assert_deployed()

    def test_rollback_after_install_before_manifest(self):
        with self.assertRaises(Interrupted):
            self.tx('after_install_rename').deploy()
        self.tx().rollback()
        self.assert_original()

    def test_resume_rollback_after_new_retired_before_manifest(self):
        self.tx().deploy()
        with self.assertRaises(Interrupted):
            self.tx('after_retire_rename').rollback()
        self.tx().rollback()
        self.assert_original()

    def test_resume_rollback_after_original_restored_before_manifest(self):
        self.tx().deploy()
        with self.assertRaises(Interrupted):
            self.tx('after_restore_rename').rollback()
        self.tx().rollback()
        self.assert_original()

    def test_running_instance_guard_before_any_mutation(self):
        def running(): raise RuntimeError('application running')
        with self.assertRaisesRegex(RuntimeError, 'running'):
            self.tx(guard=running).deploy()
        self.assert_original()
        self.assertFalse((self.home / mod.BACKUP_NAME).exists())

    def test_running_instance_guard_rechecked_before_rename(self):
        calls = 0
        def running():
            nonlocal calls
            calls += 1
            if calls >= 2: raise RuntimeError('application restarted')
        with self.assertRaisesRegex(RuntimeError, 'restarted'):
            self.tx(guard=running).deploy()
        self.assert_original()
        self.tx().rollback()
        self.assert_original()

    def test_user_edit_blocks_full_rollback_preflight(self):
        tx = self.tx()
        tx.deploy()
        self.put(self.root, 'config_client.py', b'user-edited-after-deploy')
        before = mod.snapshot(self.root)
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            self.tx().rollback()
        self.assertEqual(mod.snapshot(self.root), before)
        self.assertTrue((tx.backup / 'original/core/main.py').exists())

    def test_user_source_edit_still_blocks_rollback(self):
        self.tx().deploy()
        self.put(self.root, 'core/main.py', b'user-source-change')
        current = mod.snapshot(self.root)
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            self.tx().rollback()
        self.assertEqual(mod.snapshot(self.root), current)

    def test_prepared_user_edit_blocks_before_first_original_move(self):
        tx = self.tx()
        tx.prepare()
        self.put(self.root, 'config_client.py', b'user-edited-after-prepare')
        before = mod.snapshot(self.root)
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            self.tx().deploy()
        self.assertEqual(mod.snapshot(self.root), before)
        self.assertFalse((tx.backup / 'original/core').exists())

    def test_staged_corruption_not_overwritten(self):
        tx = self.tx()
        tx.prepare()
        self.put(tx.backup, 'prepared/core/main.py', b'corrupt-stage')
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            self.tx().deploy()
        self.assert_original()

    def test_original_backup_corruption_blocks_rollback(self):
        tx = self.tx()
        tx.deploy()
        self.put(tx.backup, 'original/models/original-model.bin', b'corrupt-backup')
        current = mod.snapshot(self.root)
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            self.tx().rollback()
        self.assertEqual(mod.snapshot(self.root), current)

    def test_path_and_policy_guards(self):
        for value in ('../escape', '/absolute', 'C:/drive', 'core/../outside', 'core\\x', 'NUL.txt', 'core./file'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                mod.relative(value)
        wrong = dict(self.plan, root=str(self.home / 'another'))
        with self.assertRaises(ValueError):
            mod.Transaction(wrong, expected_root=self.root, source_root=self.source)
        wrong = dict(self.plan, backup=str(self.root / 'backup'))
        with self.assertRaises(ValueError):
            mod.Transaction(wrong, expected_root=self.root, source_root=self.source)

    def test_component_overlap_rejected(self):
        plan = dict(self.plan, components=[{'path': 'core', 'source': 'core'}, {'path': 'core/main.py', 'source': None}])
        with self.assertRaises(ValueError):
            mod.Transaction(plan, expected_root=self.root, source_root=self.source)

    def test_plan_mutation_rejected_on_resume(self):
        self.tx().prepare()
        self.plan['components'][0]['source'] = 'models'
        with self.assertRaisesRegex(RuntimeError, 'different'):
            self.tx().deploy()
        self.assert_original()

    def test_empty_initial_manifest_pending_is_preserved_and_resumable(self):
        backup = self.home / mod.BACKUP_NAME
        backup.mkdir()
        orphan = backup / ('.manifest-' + 'a' * 32 + '.pending')
        orphan.write_text('partial interrupted write', 'utf-8')
        self.tx().deploy()
        self.assertTrue(orphan.exists())
        self.assert_deployed()

    def test_unknown_backup_refused(self):
        backup = self.home / mod.BACKUP_NAME
        self.put(backup, 'unrelated.txt', b'preserve-me')
        with self.assertRaisesRegex(RuntimeError, 'without a manifest'):
            self.tx().deploy()
        self.assert_original()
        self.assertEqual((backup / 'unrelated.txt').read_bytes(), b'preserve-me')

    def test_runtime_pycache_generation_allows_rollback_and_is_preserved(self):
        self.put(self.root, 'core/__pycache__/original.cpython-313.pyc', b'old-runtime-cache')
        self.original = mod.snapshot(self.root)
        self.put(self.source, 'core/__pycache__/source.cpython-313.pyc', b'source-cache-not-deployed')
        self.put(self.source, 'internal/gguf/reader.pyc', b'bare-bytecode-must-deploy')
        tx = self.tx()
        tx.deploy()
        self.assertFalse((self.root / 'core/__pycache__').exists())
        self.assertEqual((self.root / 'internal/gguf/reader.pyc').read_bytes(), b'bare-bytecode-must-deploy')
        import py_compile
        generated = Path(py_compile.compile(str(self.root / 'core/main.py'), doraise=True))
        generated_bytes = generated.read_bytes()
        self.assertEqual(generated_bytes[:4], importlib.util.MAGIC_NUMBER)
        tx.rollback()
        self.assert_original()
        self.assertEqual((self.root / 'core/__pycache__/original.cpython-313.pyc').read_bytes(), b'old-runtime-cache')
        self.assertEqual((tx.backup / 'retired' / generated.relative_to(self.root)).read_bytes(), generated_bytes)

    def test_pycache_directory_reparse_point_is_rejected(self):
        outside = self.home / 'outside-cache'
        outside.mkdir()
        (outside / 'cache.pyc').write_bytes(b'never-followed')
        junction = self.source / 'core/__pycache__'
        if os.name == 'nt':
            process = subprocess.run(['cmd.exe', '/c', 'mklink', '/J', str(junction), str(outside)], capture_output=True)
            if process.returncode:
                self.skipTest('Junction creation unavailable in sandbox')
        else:
            junction.symlink_to(outside, target_is_directory=True)
        try:
            with self.assertRaisesRegex(RuntimeError, 'Links/reparse'):
                self.tx().deploy()
            self.assert_original()
            self.assertEqual((outside / 'cache.pyc').read_bytes(), b'never-followed')
        finally:
            if os.name == 'nt': os.rmdir(junction)
            else: junction.unlink()

    def test_junction_source_rejected_without_following(self):
        outside = self.home / 'outside-data'
        outside.mkdir()
        (outside / 'secret.bin').write_bytes(b'outside-never-touched')
        junction = self.source / 'core/link'
        if os.name == 'nt':
            process = subprocess.run(['cmd.exe', '/c', 'mklink', '/J', str(junction), str(outside)], capture_output=True)
            if process.returncode:
                self.skipTest('Junction creation unavailable in sandbox')
        else:
            junction.symlink_to(outside, target_is_directory=True)
        try:
            with self.assertRaisesRegex(RuntimeError, 'Links/reparse'):
                self.tx().deploy()
            self.assert_original()
            self.assertEqual((outside / 'secret.bin').read_bytes(), b'outside-never-touched')
        finally:
            if os.name == 'nt': os.rmdir(junction)
            else: junction.unlink()



if __name__ == '__main__':
    unittest.main(verbosity=2)
