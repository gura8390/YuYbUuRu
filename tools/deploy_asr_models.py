"""Pinned optional-ASR deployment; default only validates an explicit ready plan.

The caller prepares and reviews shortcut targets before creating the plan. This
tool copies their pinned bytes and never executes a shortcut or stops a process.
No native DLL, old model, hotword, or other existing setting may be changed.
"""
from pathlib import Path
import argparse
import json

import deployment_transaction as txm
from deploy_encoder import absolute, sha, read_json

BACKUP_NAME = 'model-options-backup-20261027'
MODEL_DIR = 'models/Qwen3-ASR/Qwen3-ASR-1.7B'
SHORTCUTS = ('语音输入-0.6B.lnk', '语音输入-1.7B.lnk')
COMPONENTS = ('start_client.exe', 'start_server.exe', 'settings.json', MODEL_DIR, *SHORTCUTS)
NEW_COMPONENTS = (MODEL_DIR, *SHORTCUTS)
SETTINGS_PATCH = {'asr_model': '0.6b'}
MODEL_FILES = {
    'qwen3_asr_encoder_frontend.onnx': (24063978, 'fc288ba9a0ca732251b33628c803cd5188d3dd77a2cd7336f0f0fd37969ed4cd'),
    'qwen3_asr_encoder_backend.onnx': (611276553, '4b0b1e34e8393f29222006aae0719283b0ff28b61f289b6ff8377b3ddbda27e3'),
    'qwen3_asr_llm.gguf': (1471800896, '89e198470f41d487ab5b6b8a36a35400b3b1534451c4499236983050fe9193e5'),
}
NATIVE_DIR = 'core/server/engines/llama/bin'


def model_snapshot():
    return {'kind': 'directory', 'directories': [''], 'files': [
        {'path': name, 'bytes': size, 'sha256': digest}
        for name, (size, digest) in sorted(MODEL_FILES.items())]}


def file_snapshot(value):
    if (not isinstance(value, dict) or set(value) != {'kind', 'bytes', 'sha256'}
            or value['kind'] != 'file' or type(value['bytes']) is not int or value['bytes'] <= 0):
        raise ValueError('Expected a nonempty, fully pinned file snapshot')
    sha(value['sha256'])


def check_policy(plan, app_dir, source_dir, candidate_sha256):
    app_dir, source_dir = absolute(app_dir), absolute(source_dir)
    candidate_sha256 = sha(candidate_sha256)
    if plan.get('schema') != 1 or plan.get('status') != 'ready':
        raise ValueError('Only a finalized ready model-options plan is eligible')
    if absolute(plan['root']) != app_dir or absolute(plan['source_root']) != source_dir:
        raise ValueError('Explicit root/source differs from the reviewed plan')
    if absolute(plan['backup']) != app_dir.parent / BACKUP_NAME:
        raise ValueError('Backup differs from the independently named model-options sibling')
    rows = plan['components']
    if len(rows) != len(COMPONENTS) or {r['path'] for r in rows} != set(COMPONENTS):
        raise ValueError('Model-options policy requires exactly the six allowed components')
    by_path = {r['path']: r for r in rows}
    for key, row in by_path.items():
        if row.get('source') != key:
            raise ValueError('Source mapping must retain the exact allowed relative path')
        if key in NEW_COMPONENTS:
            if row.get('old') is not None:
                raise ValueError('New model/shortcut paths must be previously absent')
        else:
            file_snapshot(row.get('old'))
        if key == MODEL_DIR:
            if row.get('new') != model_snapshot():
                raise ValueError('Only the three unchanged original 1.7B model files are eligible')
        else:
            file_snapshot(row.get('new'))
    if any(by_path[key]['new']['sha256'] != candidate_sha256
           for key in ('start_client.exe', 'start_server.exe')):
        raise ValueError('Both launchers must equal the independently supplied final candidate SHA')
    if plan.get('settings_patch') != SETTINGS_PATCH or not isinstance(plan.get('settings_before'), dict):
        raise ValueError('Only asr_model=0.6b may be added to the complete original settings')
    protected = plan.get('protected_sha256')
    if not isinstance(protected, dict) or not protected:
        raise ValueError('The protected inventory is required')
    for key, digest in protected.items():
        txm.relative(key)
        sha(digest)
        if any(key == comp or key.startswith(comp + '/') for comp in COMPONENTS):
            raise ValueError('Protected inventory overlaps the update scope')
    # Preserve every existing native file, including DLLs and any auxiliary
    # files. The inventory is freshly verified before any transaction action.
    native = txm.contained(app_dir, app_dir / NATIVE_DIR)
    if not native.is_dir():
        raise ValueError('The existing native directory is required')
    native_snapshot = txm.snapshot(native)
    for item in native_snapshot['files']:
        if protected.get(NATIVE_DIR + '/' + item['path']) != item['sha256']:
            raise ValueError('Every existing native file must be protected, never updated')
    return {'root': str(app_dir), 'backup': str(app_dir.parent / BACKUP_NAME),
            'components': [{'path': r['path'], 'source': r['source']} for r in rows]}


def check_protected(plan, app_dir):
    for key, expected in plan['protected_sha256'].items():
        file = txm.contained(app_dir, app_dir / txm.relative(key))
        if txm.digest(file) != expected:
            raise RuntimeError(f'Protected file changed; preserving it: {key}')


def check_manifest_pins(tx, plan):
    tx._load()
    pins = {r['path']: r for r in plan['components']}
    for row in tx.data['components']:
        pin = pins[row['path']]
        if row['old'] != pin['old'] or row['new'] != pin['new']:
            raise RuntimeError('Transaction snapshot differs from the reviewed content pins')


def verify_current(plan, app_dir, source_dir, tx, *, installed=False):
    check_protected(plan, app_dir)
    if tx.manifest.exists():
        check_manifest_pins(tx, plan)
    if installed:
        for row in plan['components']:
            file = txm.contained(app_dir, app_dir / txm.relative(row['path']))
            txm.require_snapshot(file, row['new'])
        settings = read_json(app_dir / 'settings.json')
    else:
        if not tx.manifest.exists():
            for row in plan['components']:
                file = txm.contained(app_dir, app_dir / txm.relative(row['path']))
                txm.require_snapshot(file, row['old'])
            if read_json(app_dir / 'settings.json') != plan['settings_before']:
                raise RuntimeError('Current settings changed after plan creation')
        for row in plan['components']:
            file = txm.contained(source_dir, source_dir / txm.relative(row['source']))
            txm.require_snapshot(file, row['new'])
        settings = read_json(source_dir / 'settings.json')
    if settings != {**plan['settings_before'], **SETTINGS_PATCH}:
        raise RuntimeError('Candidate settings must preserve every original key and change only model choice')


def build_plan(app_dir, source_dir, candidate_sha256, protected_sha256):
    """Return a ready, content-pinned plan; write/stage nothing and stop no process."""
    app_dir, source_dir = absolute(app_dir), absolute(source_dir)
    rows = []
    for key in COMPONENTS:
        active = txm.contained(app_dir, app_dir / txm.relative(key))
        staged = txm.contained(source_dir, source_dir / txm.relative(key))
        rows.append({'path': key, 'source': key, 'old': txm.snapshot(active),
                     'new': txm.snapshot(staged)})
    plan = {'schema': 1, 'status': 'ready', 'root': str(app_dir),
            'source_root': str(source_dir), 'backup': str(app_dir.parent / BACKUP_NAME),
            'components': rows, 'settings_before': read_json(app_dir / 'settings.json'),
            'settings_patch': SETTINGS_PATCH.copy(), 'protected_sha256': dict(protected_sha256)}
    tx_plan = check_policy(plan, app_dir, source_dir, candidate_sha256)
    tx = txm.Transaction(tx_plan, expected_root=app_dir, source_root=source_dir, backup_name=BACKUP_NAME)
    verify_current(plan, app_dir, source_dir, tx)
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--app-dir', type=absolute, required=True)
    parser.add_argument('--source-dir', type=absolute, required=True)
    parser.add_argument('--plan', type=absolute, required=True)
    parser.add_argument('--candidate-sha256', type=sha, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--prepare-only', action='store_true')
    modes.add_argument('--execute', action='store_true')
    modes.add_argument('--verify-installed', action='store_true')
    args = parser.parse_args(argv)
    plan = read_json(args.plan)
    tx_plan = check_policy(plan, args.app_dir, args.source_dir, args.candidate_sha256)
    tx = None
    def checkpoint(event, row=None):
        if event == 'after_prepared':
            check_manifest_pins(tx, plan)
            check_protected(plan, args.app_dir)
    tx = txm.Transaction(tx_plan, expected_root=args.app_dir, source_root=args.source_dir,
                         backup_name=BACKUP_NAME, checkpoint=checkpoint)
    verify_current(plan, args.app_dir, args.source_dir, tx, installed=args.verify_installed)
    if args.verify_installed:
        print(json.dumps({'installed_hashes_and_protected_files_equal': True, 'mutated_application': False}))
    elif args.execute:
        tx.deploy()
        verify_current(plan, args.app_dir, args.source_dir, tx, installed=True)
        print(json.dumps({'deployed': True, 'protected_files_equal': True,
                          'rollback_entry': BACKUP_NAME + '/rollback_encoder.py'}))
    elif args.prepare_only:
        tx.prepare()
        print(json.dumps({'prepared': True, 'mutated_application': False}))
    else:
        print(json.dumps(tx.check(), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
