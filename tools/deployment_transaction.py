"""Resumable same-volume file transaction with self-contained rollback.

The Transaction implementation is cloned from the tested local maintenance
engine. Run deploy_encoder.py for an explicit, hash-pinned encoder policy.
No machine paths are stored in this source; generated recovery entries live
only in the private sibling backup.
"""
from pathlib import Path, PurePosixPath
from contextlib import contextmanager
import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import uuid

BACKUP_NAME = 'encoder-backup-20261027'
MANIFEST_NAME = 'manifest.json'


def backup_component_name(value):
    """A separately pinned sibling name, never a path taken from a manifest."""
    name = relative(value)
    if len(name.parts) != 1:
        raise ValueError('Backup name must be one normalized path component')
    return name.name


def relative(value):
    if not isinstance(value, str) or not value or '\\' in value or ':' in value:
        raise ValueError(f'Expected a normalized relative path: {value!r}')
    parts = PurePosixPath(value).parts
    if not parts or value.startswith('/') or any(part in ('', '.', '..') for part in value.split('/')):
        raise ValueError(f'Unsafe relative path: {value!r}')
    if any(part.rstrip(' .') != part for part in parts):
        raise ValueError(f'Ambiguous Windows path: {value!r}')
    reserved = {'CON', 'PRN', 'AUX', 'NUL', *[f'COM{i}' for i in range(1, 10)], *[f'LPT{i}' for i in range(1, 10)]}
    if any(any(c in part for c in '<>"|?*\0') or part.split('.')[0].upper() in reserved for part in parts):
        raise ValueError(f'Reserved Windows path: {value!r}')
    return Path(*parts)


def reject_link(file):
    info = file.lstat()
    if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
        raise RuntimeError(f'Links/reparse points are not permitted: {file}')
    if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
        raise RuntimeError(f'Unsupported filesystem object: {file}')


def present(file):
    try:
        file.lstat()
        return True
    except FileNotFoundError:
        return False


def contained(base, file):
    """Check lexical boundary plus every existing node, without following links."""
    base, file = Path(os.path.abspath(base)), Path(os.path.abspath(file))
    if not file.is_relative_to(base):
        raise RuntimeError(f'Path escaped its boundary: {file}')
    # A junction above base can redirect the entire named policy. Inspect each
    # ancestor with lstat, including ancestors outside the lexical base, without
    # enumerating their contents or following a reparse point.
    for node in reversed(file.parents):
        if present(node):
            reject_link(node)
    if present(file):
        reject_link(file)
    return file


def digest(file):
    reject_link(file)
    with file.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def snapshot(file):
    if not present(file):
        return None
    reject_link(file)
    if file.is_file():
        return {'kind': 'file', 'bytes': file.stat().st_size, 'sha256': digest(file)}
    directories, files = [''], []
    for folder, children, names in os.walk(file, followlinks=False):
        folder = Path(folder)
        reject_link(folder)
        runtime_children = []
        for name in sorted(children):
            child = folder / name
            reject_link(child)
            # Python source entry points may legitimately create bytecode after
            # deployment. Ignore only this cache directory, after rejecting a
            # reparse point at its root; never follow its contents. Bare .pyc
            # modules in the frozen runtime remain part of the verified tree.
            if name == '__pycache__':
                continue
            runtime_children.append(name)
            directories.append(child.relative_to(file).as_posix())
        children[:] = runtime_children
        for name in sorted(names):
            child = folder / name
            reject_link(child)
            files.append({'path': child.relative_to(file).as_posix(),
                          'bytes': child.stat().st_size, 'sha256': digest(child)})
    return {'kind': 'directory', 'directories': sorted(directories),
            'files': sorted(files, key=lambda row: row['path'])}


def require_snapshot(file, expected):
    actual = snapshot(file)
    if actual != expected:
        raise RuntimeError(f'Contents changed; preserving this path without overwriting: {file}')


def atomic_json(file, value):
    contained(file.parent, file)
    temporary = file.with_name('.manifest-' + uuid.uuid4().hex + '.pending')
    with temporary.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, file)


@contextmanager
def transaction_lock(backup):
    backup.mkdir(exist_ok=True)
    lock = contained(backup, backup / '.transaction.lock')
    with lock.open('a+b') as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError('Another deployment/rollback is running') from error
            unlock = lambda: (stream.seek(0), msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1))
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            unlock = lambda: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        try:
            yield
        finally:
            unlock()


def assert_not_running(root):
    if os.name != 'nt':
        raise RuntimeError('The production process guard requires Windows')
    command = ('[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false); '
               'Get-CimInstance Win32_Process -ErrorAction Stop | '
               'Select-Object Name,ProcessId,ExecutablePath | ConvertTo-Json -Compress')
    result = subprocess.run(['powershell.exe', '-NoProfile', '-Command', command],
        capture_output=True, check=True, creationflags=0x08000000, timeout=30)
    rows = json.loads(result.stdout.decode('utf-8-sig'))
    rows = rows if isinstance(rows, list) else [rows]
    for row in rows:
        executable = row.get('ExecutablePath')
        if executable and Path(os.path.abspath(executable)).is_relative_to(root):
            raise RuntimeError(f'Exit the application before deployment/rollback: PID {row["ProcessId"]}')
        if not executable and row.get('Name', '').lower() in ('start_client.exe', 'start_server.exe'):
            raise RuntimeError('An application launcher is running but its location cannot be verified')


class Transaction:
    def __init__(self, plan, *, expected_root, source_root, guard=None, checkpoint=None,
                 backup_name=BACKUP_NAME):
        self.root = Path(os.path.abspath(expected_root))
        self.backup_name = backup_component_name(backup_name)
        self.backup = self.root.parent / self.backup_name
        self.source = Path(os.path.abspath(source_root))
        if Path(os.path.abspath(plan['root'])) != self.root or Path(os.path.abspath(plan['backup'])) != self.backup:
            raise ValueError('Plan root/backup does not match the independently named policy')
        if self.root == self.backup or self.backup.is_relative_to(self.root):
            raise ValueError('Backup must remain outside the application')
        if self.root.drive.lower() != self.backup.drive.lower():
            raise ValueError('Backup must be on the same volume')
        self.plan = {'root': str(self.root), 'backup': str(self.backup), 'components': []}
        seen = []
        for row in plan['components']:
            key = relative(row['path'])
            if any(key == prior or key.is_relative_to(prior) or prior.is_relative_to(key) for prior in seen):
                raise ValueError(f'Duplicate/overlapping deployment component: {key}')
            seen.append(key)
            source = row.get('source')
            if source is not None:
                relative(source)
            self.plan['components'].append({'path': key.as_posix(), 'source': source})
        if not seen:
            raise ValueError('No deployment components')
        self.guard = guard or (lambda: assert_not_running(self.root))
        self.checkpoint = checkpoint or (lambda event, row=None: None)
        self.manifest = self.backup / MANIFEST_NAME
        self.data = None
        self._boundaries()

    def _boundaries(self):
        for base in (self.root, self.source, self.backup):
            # Include any existing parent at the named policy boundary.
            contained(base.parent, base)
        for row in self.plan['components']:
            contained(self.root, self.root / relative(row['path']))
            if row['source'] is not None:
                contained(self.source, self.source / relative(row['source']))
            for branch in ('original', 'prepared', 'retired'):
                contained(self.backup, self.backup / branch / relative(row['path']))

    def _paths(self, row):
        key = relative(row['path'])
        return tuple(contained(base, base / key) for base in
                     (self.root, self.backup / 'original', self.backup / 'prepared', self.backup / 'retired'))

    def _save(self):
        self._boundaries()
        atomic_json(self.manifest, self.data)

    def _load(self):
        reject_link(self.manifest)
        self.data = json.loads(self.manifest.read_text('utf-8'))
        if self.data.get('version') != 1 or self.data['plan'] != self.plan or self.data['source_root'] != str(self.source):
            raise RuntimeError('Existing transaction belongs to a different plan/policy')
        if [{'path': row['path'], 'source': row['source']} for row in self.data['components']] != self.plan['components']:
            raise RuntimeError('Manifest components disagree with the verified plan')

    def _helpers(self):
        engine = Path(__file__).read_bytes()
        wrapper = ('"""Prepared before deployment; defaults to validation only."""\n'
                   'from pathlib import Path\nfrom deployment_transaction import backup_rollback_cli\n'
                   'if __name__ == "__main__":\n'
                   f'    backup_rollback_cli(Path({str(self.root)!r}), Path(__file__).resolve().parent, '
                   f'Path({str(self.source)!r}), {self.plan["components"]!r}, '
                   f'backup_name={self.backup_name!r})\n').encode('utf-8')
        for name, payload in (('deployment_transaction.py', engine), ('rollback_encoder.py', wrapper)):
            file = contained(self.backup, self.backup / name)
            if file.exists():
                if file.read_bytes() != payload:
                    raise RuntimeError(f'Existing recovery helper changed: {file}')
            else:
                temporary = self.backup / ('.helper-' + uuid.uuid4().hex + '.pending')
                with temporary.open('xb') as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.rename(temporary, file)

    def check(self):
        self._boundaries()
        if not self.root.is_dir():
            raise RuntimeError('The named original application directory is missing')
        self.guard()
        return {'root': str(self.root), 'backup': str(self.backup), 'components': len(self.plan['components']),
                'existing_transaction': self.manifest.exists(), 'mutated': False}

    def _copy_file(self, source, destination, expected, row):
        contained(self.backup, destination)
        if destination.exists():
            require_snapshot(destination, expected)
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        incoming = contained(self.backup, self.backup / 'incoming')
        incoming.mkdir(exist_ok=True)
        temporary = incoming / (uuid.uuid4().hex + '.part')
        contained(self.source, source)
        reject_link(source)
        with source.open('rb') as incoming_stream, temporary.open('xb') as staged_stream:
            shutil.copyfileobj(incoming_stream, staged_stream, length=8 * 1024 * 1024)
            staged_stream.flush()
            os.fsync(staged_stream.fileno())
        shutil.copystat(source, temporary, follow_symlinks=False)
        require_snapshot(temporary, expected)
        os.rename(temporary, destination)
        self.checkpoint('after_stage_file', row)

    def _stage(self, row):
        if row['new'] is None:
            return
        source = contained(self.source, self.source / relative(row['source']))
        target = self._paths(row)[2]
        require_snapshot(source, row['new'])
        if row['new']['kind'] == 'file':
            self._copy_file(source, target, row['new'], row)
        else:
            for name in row['new']['directories']:
                folder = target if name == '' else target / relative(name)
                contained(self.backup, folder)
                folder.mkdir(parents=True, exist_ok=True)
            for file in row['new']['files']:
                key = relative(file['path'])
                expected = {k: v for k, v in file.items() if k != 'path'}
                expected['kind'] = 'file'
                self._copy_file(source / key, target / key, expected, row)
        require_snapshot(target, row['new'])
        self.checkpoint('after_prepare_component', row)

    def _prepare(self):
        if self.manifest.exists():
            self._load()
            if self.data['phase'] != 'preparing':
                return
        else:
            unknown = [p.name for p in self.backup.iterdir() if p.name != '.transaction.lock'
                       and re.fullmatch(r'\.manifest-[0-9a-f]{32}\.pending', p.name) is None]
            if unknown:
                raise RuntimeError(f'Backup exists without a manifest; refusing to reuse: {unknown}')
            rows = []
            for item in self.plan['components']:
                old = snapshot(self.root / relative(item['path']))
                new = snapshot(self.source / relative(item['source'])) if item['source'] is not None else None
                if item['source'] is not None and new is None:
                    raise RuntimeError(f'Missing staged source: {item["source"]}')
                rows.append({**item, 'old': old, 'new': new, 'state': 'preparing'})
            self.data = {'version': 1, 'transaction_id': uuid.uuid4().hex, 'plan': self.plan,
                         'source_root': str(self.source), 'phase': 'preparing', 'components': rows}
            self._save()
        self._helpers()
        for row in self.data['components']:
            self._stage(row)
            row['state'] = 'prepared'
            self._save()
        self.data['phase'] = 'prepared'
        self._save()
        self.checkpoint('after_prepared')

    def prepare(self):
        self.check()
        with transaction_lock(self.backup):
            self._prepare()

    def _intent(self, row, state):
        row['state'] = state
        self._save()
        self.guard()

    def _rename(self, source, target):
        self._boundaries()
        if not source.exists() or target.exists() or target.is_symlink():
            raise RuntimeError(f'Rename would lose or overwrite data: {source} -> {target}')
        target.parent.mkdir(parents=True, exist_ok=True)
        self._boundaries()
        os.rename(source, target)

    def _deploy_row(self, row):
        active, archived, prepared, retired = self._paths(row)
        if retired.exists():
            raise RuntimeError('Transaction already started rollback')
        if archived.exists():
            if row['old'] is None:
                raise RuntimeError(f'Unexpected original backup: {archived}')
            require_snapshot(archived, row['old'])
        elif row['old'] is not None:
            require_snapshot(active, row['old'])
            self._intent(row, 'saving_original')
            self._rename(active, archived)
            self.checkpoint('after_archive_rename', row)
            require_snapshot(archived, row['old'])
        if row['new'] is not None:
            if active.exists():
                if prepared.exists():
                    raise RuntimeError(f'Active destination was created/edited after preparation: {active}')
                require_snapshot(active, row['new'])
            else:
                require_snapshot(prepared, row['new'])
                self._intent(row, 'installing')
                self._rename(prepared, active)
                self.checkpoint('after_install_rename', row)
                require_snapshot(active, row['new'])
            row['state'] = 'installed'
        else:
            require_snapshot(active, None)
            row['state'] = 'archived'
        self._save()

    def _deploy_preflight(self):
        for row in self.data['components']:
            active, archived, prepared, retired = self._paths(row)
            if retired.exists():
                raise RuntimeError('Transaction already started rollback')
            old, new, current, saved, staged = row['old'], row['new'], snapshot(active), snapshot(archived), snapshot(prepared)
            if saved is not None:
                if old is None or saved != old:
                    raise RuntimeError(f'Original backup changed: {archived}')
                if current is not None and (new is None or current != new or staged is not None):
                    raise RuntimeError(f'Active destination changed: {active}')
            elif old is not None:
                if current != old:
                    raise RuntimeError(f'Original destination changed after preparation: {active}')
            elif current is not None and (new is None or current != new or staged is not None):
                raise RuntimeError(f'New destination changed after preparation: {active}')
            if new is not None:
                # Either staged and not installed, or the active tree is the new one.
                if staged is not None:
                    if staged != new:
                        raise RuntimeError(f'Prepared component changed: {prepared}')
                elif current != new:
                    raise RuntimeError(f'Prepared component is missing: {prepared}')

    def deploy(self):
        self.check()
        with transaction_lock(self.backup):
            self._prepare()
            if self.data['phase'] in ('rolling_back', 'rolled_back'):
                raise RuntimeError('A rolled-back transaction cannot be redeployed; keep the backup')
            self._helpers()  # Recovery entry must exist before the first active rename.
            self._deploy_preflight()
            self.data['phase'] = 'deploying'
            self._save()
            for row in self.data['components']:
                self._deploy_row(row)
            self.data['phase'] = 'deployed'
            self._save()

    def _rollback_state(self, row):
        active, archived, prepared, retired = self._paths(row)
        a, o, n = snapshot(active), snapshot(archived), snapshot(retired)
        if o is not None and (row['old'] is None or o != row['old']):
            raise RuntimeError(f'Original backup changed: {archived}')
        if n is not None and (row['new'] is None or n != row['new']):
            raise RuntimeError(f'Retired component changed: {retired}')
        if o is not None:
            if a is not None and (row['new'] is None or a != row['new'] or n is not None):
                raise RuntimeError(f'Active component changed after deployment: {active}')
            return 'restore', a is not None
        if row['old'] is not None:
            if a != row['old']:
                raise RuntimeError(f'Original data is missing/changed; refusing rollback: {active}')
            return 'done', False
        if a is not None and (row['new'] is None or a != row['new'] or n is not None):
            raise RuntimeError(f'New destination changed after deployment: {active}')
        return 'remove_new', a is not None

    def rollback(self):
        self.check()
        with transaction_lock(self.backup):
            self._load()
            # Reject all subsequent user edits before moving any active component.
            for row in self.data['components']:
                self._rollback_state(row)
            self.data['phase'] = 'rolling_back'
            self._save()
            for row in reversed(self.data['components']):
                mode, retire = self._rollback_state(row)
                active, archived, prepared, retired = self._paths(row)
                if retire:
                    self._intent(row, 'retiring_new')
                    self._rename(active, retired)
                    self.checkpoint('after_retire_rename', row)
                if mode == 'restore':
                    self._intent(row, 'restoring_original')
                    self._rename(archived, active)
                    self.checkpoint('after_restore_rename', row)
                    require_snapshot(active, row['old'])
                row['state'] = 'rolled_back'
                self._save()
            self.data['phase'] = 'rolled_back'
            self._save()


def pinned_component_map(rows):
    """Normalize a manifest list while rejecting duplicate/ambiguous scope."""
    if not isinstance(rows, list) or not rows:
        raise RuntimeError('Recovery component pin must be a nonempty list')
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError('Recovery component record must be an object')
        path = relative(row['path']).as_posix()
        source = row.get('source')
        if source is not None:
            source = relative(source).as_posix()
        if path in result:
            raise RuntimeError('Duplicate recovery component is outside the original scope')
        result[path] = source
    return result


def check_recovery_pins(data, expected_root, backup, source_root, expected_components):
    """Pins come from the generated helper, never from mutable manifest data."""
    plan = data['plan']
    if (Path(os.path.abspath(plan['root'])) != Path(os.path.abspath(expected_root))
            or Path(os.path.abspath(plan['backup'])) != Path(os.path.abspath(backup))
            or Path(os.path.abspath(data['source_root'])) != Path(os.path.abspath(source_root))):
        raise RuntimeError('Recovery root/source differs from the original helper pins')
    expected = pinned_component_map(expected_components)
    planned = pinned_component_map(plan['components'])
    actual = pinned_component_map(data['components'])
    if (len(plan['components']) != len(expected_components)
            or len(data['components']) != len(expected_components)
            or planned != expected or actual != expected):
        raise RuntimeError('Recovery components differ from the original helper pins')


def backup_rollback_cli(expected_root, backup, source_root, expected_components,
                        *, backup_name=BACKUP_NAME):
    backup_name = backup_component_name(backup_name)
    if Path(os.path.abspath(backup)) != Path(os.path.abspath(expected_root)).parent / backup_name:
        raise RuntimeError('Recovery entry is outside its named sibling backup')
    contained(backup.parent, backup)
    manifest = contained(backup, backup / MANIFEST_NAME)
    data = json.loads(manifest.read_text('utf-8'))
    check_recovery_pins(data, expected_root, backup, source_root, expected_components)
    tx = Transaction(data['plan'], expected_root=expected_root, source_root=source_root,
                     backup_name=backup_name)
    parser = argparse.ArgumentParser(description='Restore the original application without deleting retired data')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.execute:
        tx.rollback()
        print('Rollback complete; new/old data retained in the sibling backup.')
    else:
        tx.check()
        tx._load()
        for row in tx.data['components']:
            tx._rollback_state(row)
        print('Rollback preflight passed; use --execute to restore.')



if __name__ == '__main__':
    raise SystemExit('Use deploy_encoder.py with explicit app-dir, source-dir and private plan paths.')
