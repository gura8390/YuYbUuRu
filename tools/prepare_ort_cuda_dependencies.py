"""Stage only approved cuDNN loader/cuFFT members; static PE checks, no CUDA or ORT initialization."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import time
import urllib.request

from prepare_ort_cuda_runtime import zip_members, extract_member, dump_dependencies

FRONTEND_ENGINES = ('cudnn_engines_precompiled64_9.dll',
                    'cudnn_engines_runtime_compiled64_9.dll', 'cudnn_heuristic64_9.dll')


def sha_file(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_receipt(path, value):
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, indent=2) + '\n', 'utf-8')
    temporary.replace(path)


def prepare_frontend_engines(args, output):
    """Explicit phase after an actual frontend Conv reported a missing engine."""
    base_path = output / 'dependency-preparation.json'
    base = json.loads(base_path.read_text('utf-8'))
    if not base.get('prepared') or base['cudnn']['version'] != '9.7.1.26':
        raise RuntimeError('Matching base dependencies must already be prepared')
    for row in base['cudnn']['selected_members']:
        if sha_file(output / row['filename']) != row['sha256']:
            raise RuntimeError('Previously verified cuDNN DLL changed')
    destination = output / 'frontend-engine-preparation.json'
    if destination.exists():
        receipt = json.loads(destination.read_text('utf-8'))
        if receipt.get('phase') != 'frontend-engines' or receipt.get('version') != '9.7.1.26':
            raise RuntimeError('Unknown existing frontend-engine receipt')
    else:
        receipt = {'phase': 'frontend-engines', 'version': '9.7.1.26', 'prepared': False,
                   'started_unix_seconds': time.time(), 'gpu_initialized': False,
                   'sessions_created': 0, 'models_loaded': 0,
                   'system_install_or_formal_directory_changes': False,
                   'base_dependency_receipt_sha256': sha_file(base_path), 'selected_members': [],
                   'approved_names': list(FRONTEND_ENGINES), 'adv_downloaded': False}
    metadata_url = 'https://pypi.org/pypi/nvidia-cudnn-cu12/9.7.1.26/json'
    metadata = json_url(metadata_url)
    wheel = next(row for row in metadata['urls'] if row['filename'].endswith('win_amd64.whl'))
    members = zip_members(wheel['url'], wheel['size'])
    entries = {Path(row['name']).name: row for row in members.values()}
    receipt.update(metadata_url=metadata_url, wheel_url=wheel['url'], wheel_bytes=wheel['size'],
                   wheel_sha256_published_not_full_download_verified=wheel['digests']['sha256'],
                   selected_compressed_bytes=sum(entries[name]['compressed_bytes'] for name in FRONTEND_ENGINES),
                   selected_expanded_bytes=sum(entries[name]['bytes'] for name in FRONTEND_ENGINES))
    completed = {row['filename']: row for row in receipt['selected_members']}
    if not set(completed).issubset(FRONTEND_ENGINES):
        raise RuntimeError('Existing receipt contains an unapproved engine')
    for name in FRONTEND_ENGINES:
        target = output / name
        if target.exists() and (name not in completed or sha_file(target) != completed[name]['sha256']):
            raise RuntimeError(f'Refusing to overwrite an existing DLL: {name}')
        if name in completed and not target.exists():
            raise RuntimeError(f'Previously verified engine is missing: {name}')
    save_receipt(destination, receipt)
    print(json.dumps({'phase': 'frontend-engines', 'compressed_bytes': receipt['selected_compressed_bytes'],
                      'expanded_bytes': receipt['selected_expanded_bytes'], 'already_verified': list(completed)}), flush=True)
    try:
        for name in FRONTEND_ENGINES:
            if name in completed:
                continue
            receipt['active_member'] = name
            partial = (output / name).with_suffix('.dll.part')
            if partial.exists():
                receipt.setdefault('aborted_partials', []).append({'filename': partial.name,
                    'bytes': partial.stat().st_size, 'sha256': sha_file(partial)})
            save_receipt(destination, receipt)
            row = extract_member(wheel['url'], entries[name], output / name)
            receipt['selected_members'].append(row)
            save_receipt(destination, receipt)
            print(json.dumps({'phase': 'frontend-engine-completed', 'member': row}), flush=True)
        missing = set()
        for name in FRONTEND_ENGINES:
            for option, suffix in (('/DEPENDENTS', 'dependents'), ('/IMPORTS', 'imports')):
                result = subprocess.run([str(args.dumpbin), option, str(output / name)], check=True,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, creationflags=0x08000000)
                text = result.stdout.decode('utf-8', errors='replace')
                (output / f'{name}.{suffix}.txt').write_text(text, 'utf-8')
                if option == '/DEPENDENTS':
                    for dependency in re.findall(r'^\s+([a-zA-Z0-9_.-]+\.dll)\s*$', text, re.M | re.I):
                        if dependency.lower().startswith(('api-ms-', 'ext-ms-')):
                            continue
                        if not any((directory / dependency).is_file() for directory in
                                   (output, args.existing_cuda_dir, Path('C:/Windows/System32'))):
                            missing.add(dependency)
        receipt.update(prepared=True, active_member=None, missing_static_engine_imports=sorted(missing),
                       elapsed_seconds=time.time() - receipt['started_unix_seconds'])
        receipt.pop('error', None)
    except BaseException as error:
        receipt['error'] = repr(error)
        raise
    finally:
        save_receipt(destination, receipt)
    print(json.dumps({'prepared': True, 'phase': 'frontend-engines',
                      'missing_static_engine_imports': sorted(missing), 'models_loaded': 0}), flush=True)


def json_url(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--existing-cuda-dir', type=Path, required=True)
    parser.add_argument('--dumpbin', type=Path,
                        default=shutil.which('dumpbin.exe') or shutil.which('dumpbin'),
                        help='MSVC dumpbin; defaults to PATH in a Developer shell')
    parser.add_argument('--continue-after-loader', action='store_true')
    parser.add_argument('--encoder-sublibraries', action='store_true', help='Explicitly approved graph/ops/cnn only; never fetch adv or engines')
    parser.add_argument('--frontend-engines', action='store_true', help='Explicit second phase: only precompiled/runtime_compiled/heuristic, after a real frontend missing-engine error')
    args = parser.parse_args()
    if args.dumpbin is None:
        parser.error('dumpbin was not found on PATH; use an MSVC Developer shell or pass --dumpbin')
    output = args.output.resolve(strict=True)
    if args.frontend_engines:
        if args.continue_after_loader or args.encoder_sublibraries:
            parser.error('--frontend-engines is a separate explicit preparation phase')
        return prepare_frontend_engines(args, output)
    required = ('cudnn64_9.dll', 'cufft64_11.dll')
    if not args.continue_after_loader and (any((output / name).exists() for name in required) or (output / 'dependency-preparation.json').exists()):
        raise RuntimeError('Refusing to replace existing staged dependencies')
    metadata_url = 'https://pypi.org/pypi/nvidia-cudnn-cu12/9.7.1.26/json'
    metadata = json_url(metadata_url)
    wheel = next(row for row in metadata['urls'] if row['filename'].endswith('win_amd64.whl'))
    cudnn_members = zip_members(wheel['url'], wheel['size'])
    candidates = [entry for name, entry in cudnn_members.items() if name.endswith('.dll')]
    top = next(entry for entry in candidates if Path(entry['name']).name == 'cudnn64_9.dll')
    receipt = {'prepared': False, 'gpu_initialized': False, 'sessions_created': 0, 'models_loaded': 0,
               'system_install_or_formal_directory_changes': False,
               'cudnn': {'version': '9.7.1.26', 'metadata_url': metadata_url, 'wheel_url': wheel['url'],
                         'wheel_bytes': wheel['size'], 'selected_members': [],
                         'available_member_sizes': [{key: entry[key] for key in ('name', 'compressed_bytes', 'bytes')} for entry in candidates]},
               'cufft': {}}
    destination = output / 'dependency-preparation.json'
    if args.continue_after_loader:
        prior = json.loads(destination.read_text('utf-8'))
        if prior.get('prepared') or prior.get('cufft', {}).get('selected_members'):
            raise RuntimeError('Continuation is only valid after the loader-only interruption; refusing to overwrite verified DLLs')
        receipt['cudnn']['selected_members'] = prior['cudnn']['selected_members']
        for item in receipt['cudnn']['selected_members']:
            if hashlib.sha256((output / item['filename']).read_bytes()).hexdigest() != item['sha256']:
                raise RuntimeError('Previously staged cuDNN loader changed')
        partial = output / 'cufft64_11.dll.part'
        if partial.exists():
            receipt['aborted_redist_attempt'] = {'partial_bytes': partial.stat().st_size,
                'compressed_bytes_reported': 33554432, 'reason': 'Switched slow redist stream to official matching PyPI member',
                'url': 'https://developer.download.nvidia.com/compute/cuda/redist/libcufft/windows-x86_64/libcufft-windows-x86_64-11.3.3.83-archive.zip',
                'approved_member_compressed_bytes': 190453191,
                'partial_sha256': hashlib.sha256(partial.read_bytes()).hexdigest()}
    try:
        if not args.continue_after_loader:
            receipt['cudnn']['selected_members'].append(extract_member(wheel['url'], top, output / 'cudnn64_9.dll'))
        for option, suffix in (('/DEPENDENTS', 'dependents'), ('/IMPORTS', 'imports')):
            result = subprocess.run([str(args.dumpbin), option, str(output / 'cudnn64_9.dll')], check=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, creationflags=0x08000000)
            (output / f'cudnn64_9.dll.{suffix}.txt').write_text(result.stdout.decode('utf-8', errors='replace'), 'utf-8')
        receipt['after_cudnn_loader'] = dump_dependencies(args.dumpbin, output, args.existing_cuda_dir)
        destination.write_text(json.dumps(receipt, indent=2), 'utf-8')
        print(json.dumps({'stage': 'cudnn-loader', 'member': receipt['cudnn']['selected_members'][0],
                          'missing_dlls': receipt['after_cudnn_loader']['missing_dlls']}), flush=True)
        cufft_metadata_url = 'https://pypi.org/pypi/nvidia-cufft-cu12/11.3.3.83/json'
        cufft_metadata = json_url(cufft_metadata_url)
        artifact = next(row for row in cufft_metadata['urls'] if row['filename'].endswith('win_amd64.whl'))
        url = artifact['url']; members = zip_members(url, artifact['size'])
        entry = next(value for key, value in members.items() if Path(key).name == 'cufft64_11.dll')
        if entry['bytes'] != 276121600 or entry['crc32'] != 0xb98b4c50:
            raise RuntimeError('PyPI cuFFT does not match the approved NVIDIA redist member metadata')
        receipt['cufft'] = {'version': '11.3.3.83', 'metadata_url': cufft_metadata_url, 'artifact_url': url,
                           'archive_bytes': artifact['size'], 'archive_sha256_published_not_full_download_verified': artifact['digests']['sha256'],
                           'selected_members': [extract_member(url, entry, output / 'cufft64_11.dll')]}
        if args.encoder_sublibraries:
            for name in ('cudnn_graph64_9.dll', 'cudnn_ops64_9.dll', 'cudnn_cnn64_9.dll'):
                target = output / name
                if target.exists():
                    raise RuntimeError('Refusing to overwrite cuDNN encoder sublibrary')
                selected = next(entry for entry in candidates if Path(entry['name']).name == name)
                item = extract_member(wheel['url'], selected, target)
                receipt['cudnn']['selected_members'].append(item)
                destination.write_text(json.dumps(receipt, indent=2), 'utf-8')
                print(json.dumps({'stage': 'encoder-sublibrary', 'member': item}), flush=True)
        receipt['dependencies'] = dump_dependencies(args.dumpbin, output, args.existing_cuda_dir)
        receipt['prepared'] = True
    finally:
        destination.write_text(json.dumps(receipt, indent=2), 'utf-8')
    print(json.dumps({'prepared': True, 'missing_dlls': receipt['dependencies']['missing_dlls'],
                      'gpu_initialized': False, 'sessions_created': 0, 'models_loaded': 0}), flush=True)


if __name__ == '__main__':
    main()
