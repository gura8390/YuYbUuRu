"""Hash-pinned encoder update; default mode only checks the explicit private plan.

All app/source/plan paths are supplied by the caller. A draft plan cannot run.
Only the two launchers, settings.json and the selected llama DLL are eligible.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re

import deployment_transaction as txm

COMPONENTS = ('start_client.exe', 'start_server.exe', 'settings.json',
              'core/server/engines/llama/bin/llama.dll')
SETTINGS_PATCH = {
    'gpu_backend': 'vulkan', 'vulkan_pipeline_cache': True,
    'asr_single_output': True, 'asr_dynamic_context': False,
    'cuda_graph_optimization': False, 'encoder_provider': 'cpu',
    'encoder_frontend_arena': True, 'encoder_backend_arena': False,
    'encoder_parallel_execution': False, 'encoder_spin_wait': False,
    'encoder_dynamic_block_base': 0, 'encoder_frontend_threads': None,
    'onnx_arena_shrink': False, 'onnx_device_initializers': False,
}


def absolute(path):
    value = Path(path)
    if not value.is_absolute():
        raise ValueError('Explicit absolute paths are required')
    return Path(os.path.abspath(value))


def sha(value):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('A final lowercase SHA256 is required; draft hashes cannot execute')
    return value


def read_json(file):
    txm.contained(file.parent, file)
    return json.loads(file.read_text('utf-8'))


def check_policy(plan, app_dir, source_dir):
    app_dir, source_dir = absolute(app_dir), absolute(source_dir)
    if plan.get('schema') != 1 or plan.get('status') != 'ready':
        raise ValueError('Only a finalized ready plan can be checked or executed')
    if absolute(plan['root']) != app_dir or absolute(plan['source_root']) != source_dir:
        raise ValueError('Explicit app/source policy differs from the reviewed plan')
    if absolute(plan['backup']) != app_dir.parent / txm.BACKUP_NAME:
        raise ValueError('Backup is outside the separately named encoder sibling')
    rows = plan['components']
    if len(rows) != len(COMPONENTS) or {row['path'] for row in rows} != set(COMPONENTS):
        raise ValueError('Encoder policy requires exactly the four allowed files')
    for row in rows:
        if row.get('source') != row['path']:
            raise ValueError('Source mapping must use the same allowed relative path')
        sha(row['old_sha256'])
        sha(row['new_sha256'])
    new = {row['path']: row['new_sha256'] for row in rows}
    if new['start_client.exe'] != new['start_server.exe']:
        raise ValueError('The two launchers must contain the same final binary')
    if plan.get('settings_patch') != SETTINGS_PATCH:
        raise ValueError('Settings policy differs from the reviewed frontend-only configuration')
    if not isinstance(plan.get('settings_before'), dict):
        raise ValueError('The original settings object must be retained in the private plan')
    if not plan.get('protected_sha256'):
        raise ValueError('The protected file inventory is required')
    for key, digest in plan['protected_sha256'].items():
        txm.relative(key)
        if key in COMPONENTS:
            raise ValueError('Protected inventory overlaps an updated component')
        sha(digest)
    tx_plan = {'root': str(app_dir), 'backup': str(app_dir.parent / txm.BACKUP_NAME),
               'components': [{'path': row['path'], 'source': row['source']} for row in rows]}
    return tx_plan


def check_digest(base, relative, expected):
    file = txm.contained(base, base / txm.relative(relative))
    if txm.digest(file) != expected:
        raise RuntimeError(f'Hash mismatch; preserving file: {relative}')


def check_protected(plan, app_dir):
    for key, digest in plan['protected_sha256'].items():
        check_digest(app_dir, key, digest)


def check_manifest_pins(tx, plan):
    tx._load()
    pins = {row['path']: row for row in plan['components']}
    for row in tx.data['components']:
        old, new, pin = row['old'], row['new'], pins[row['path']]
        if (old is None or new is None or old.get('kind') != 'file' or new.get('kind') != 'file'
                or old.get('sha256') != pin['old_sha256'] or new.get('sha256') != pin['new_sha256']):
            raise RuntimeError('Transaction content differs from the approved hash pins')


def verify_current(plan, app_dir, source_dir, tx, *, installed=False):
    check_protected(plan, app_dir)
    if installed:
        for row in plan['components']:
            check_digest(app_dir, row['path'], row['new_sha256'])
        settings = read_json(app_dir / 'settings.json')
    else:
        if tx.manifest.exists():
            check_manifest_pins(tx, plan)
        else:
            for row in plan['components']:
                check_digest(app_dir, row['path'], row['old_sha256'])
            if read_json(app_dir / 'settings.json') != plan['settings_before']:
                raise RuntimeError('Current settings changed after plan creation')
        for row in plan['components']:
            check_digest(source_dir, row['source'], row['new_sha256'])
        settings = read_json(source_dir / 'settings.json')
    if settings != {**plan['settings_before'], **SETTINGS_PATCH}:
        raise RuntimeError('Candidate settings must preserve every existing key and apply only the reviewed patch')


def finalized_plan(draft, app_dir, source_dir, candidate_sha256):
    """Explicitly pin an already-built candidate. Does not stage or deploy files."""
    candidate_sha256 = sha(candidate_sha256)
    plan = dict(draft)
    plan['root'], plan['source_root'] = str(app_dir), str(source_dir)
    plan['backup'] = str(app_dir.parent / txm.BACKUP_NAME)
    plan['components'] = [dict(row) for row in draft['components']]
    for row in plan['components']:
        file = txm.contained(source_dir, source_dir / txm.relative(row['source']))
        actual = txm.digest(file)
        if row['path'] in ('start_client.exe', 'start_server.exe'):
            if actual != candidate_sha256:
                raise RuntimeError('Final launcher differs from the explicitly supplied candidate SHA256')
        elif row['new_sha256'] is not None and actual != row['new_sha256']:
            raise RuntimeError('Final DLL/settings differ from the reviewed draft')
        row['new_sha256'] = actual
    plan['status'] = 'ready'
    check_policy(plan, app_dir, source_dir)
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--app-dir', type=absolute, required=True)
    parser.add_argument('--source-dir', type=absolute, required=True)
    parser.add_argument('--plan', type=absolute, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--execute', action='store_true')
    modes.add_argument('--prepare-only', action='store_true')
    modes.add_argument('--verify-installed', action='store_true')
    modes.add_argument('--finalize-plan', type=absolute, metavar='OUTPUT')
    parser.add_argument('--candidate-sha256')
    args = parser.parse_args(argv)
    plan = read_json(args.plan)
    if args.finalize_plan:
        if args.finalize_plan.exists():
            raise FileExistsError('Refusing to replace an existing reviewed plan')
        plan = finalized_plan(plan, args.app_dir, args.source_dir, args.candidate_sha256)
        tx_plan = check_policy(plan, args.app_dir, args.source_dir)
        tx = txm.Transaction(tx_plan, expected_root=args.app_dir, source_root=args.source_dir)
        verify_current(plan, args.app_dir, args.source_dir, tx)
        txm.atomic_json(args.finalize_plan, plan)
        print(json.dumps({'status': 'ready', 'mutated_application': False, 'files': len(COMPONENTS)}))
        return 0
    tx_plan = check_policy(plan, args.app_dir, args.source_dir)
    tx = None
    def checkpoint(event, row=None):
        if event == 'after_prepared':
            check_manifest_pins(tx, plan)
            check_protected(plan, args.app_dir)
    tx = txm.Transaction(tx_plan, expected_root=args.app_dir, source_root=args.source_dir, checkpoint=checkpoint)
    verify_current(plan, args.app_dir, args.source_dir, tx, installed=args.verify_installed)
    if args.verify_installed:
        print(json.dumps({'installed_hashes_and_protected_files_equal': True, 'mutated_application': False}))
    elif args.execute:
        tx.deploy()
        verify_current(plan, args.app_dir, args.source_dir, tx, installed=True)
        print(json.dumps({'deployed': True, 'protected_files_equal': True,
                          'rollback_entry': txm.BACKUP_NAME + '/rollback_encoder.py'}))
    elif args.prepare_only:
        tx.prepare()
        print(json.dumps({'prepared': True, 'mutated_application': False}))
    else:
        print(json.dumps(tx.check(), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
