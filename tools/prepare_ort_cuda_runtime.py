"""Selectively fetch matching official ORT CUDA DLLs via ZIP ranges; no session/model or system install."""
from pathlib import Path
import argparse
import ctypes
import hashlib
import json
import re
import shutil
import struct
import subprocess
import urllib.request
import zlib

TARGETS = ('onnxruntime.dll', 'onnxruntime_providers_shared.dll', 'onnxruntime_providers_cuda.dll')


def range_response(url, start, size):
    response = urllib.request.urlopen(urllib.request.Request(url, headers={'Range': f'bytes={start}-{start + size - 1}'}), timeout=45)
    if response.status != 206 or response.headers.get('Content-Range', '').split('/')[0] != f'bytes {start}-{start + size - 1}':
        response.close()
        raise RuntimeError('Server did not honor exact range; refusing full package')
    return response


def read_range(url, start, size):
    with range_response(url, start, size) as response:
        result = response.read(size + 1)
    if len(result) != size:
        raise RuntimeError('Range length differs')
    return result


def zip_members(url, size):
    start = max(0, size - 65536)
    tail = read_range(url, start, size - start)
    position = tail.rfind(b'PK\x05\x06')
    if position < 0:
        raise RuntimeError('ZIP end record absent')
    end = struct.unpack_from('<4s4H2LH', tail, position)
    directory_size, directory_offset = end[5], end[6]
    directory = tail[directory_offset - start:directory_offset - start + directory_size] if directory_offset >= start else read_range(url, directory_offset, directory_size)
    members, position = {}, 0
    for _ in range(end[4]):
        header = struct.unpack_from('<4s6H3L5H2L', directory, position)
        if header[0] != b'PK\x01\x02':
            raise RuntimeError('Invalid ZIP central record')
        name = directory[position + 46:position + 46 + header[10]].decode('utf-8')
        members[name] = {'name': name, 'flags': header[3], 'compression': header[4], 'crc32': header[7],
                         'compressed_bytes': header[8], 'bytes': header[9], 'local_offset': header[16]}
        position += 46 + header[10] + header[11] + header[12]
    return members


def extract_member(url, entry, output):
    if entry['compression'] != 8 or entry['flags'] & 1 or entry['bytes'] > 512 * 1024 * 1024:
        raise RuntimeError('Unexpected encrypted/compressed/oversized member')
    header = struct.unpack('<4s5H3L2H', read_range(url, entry['local_offset'], 30))
    if header[0] != b'PK\x03\x04':
        raise RuntimeError('Invalid ZIP local record')
    offset = entry['local_offset'] + 30 + header[9] + header[10]
    temporary = output.with_suffix(output.suffix + '.part')
    crc = written = consumed = 0
    digest = hashlib.sha256(); decoder = zlib.decompressobj(-15)
    with range_response(url, offset, entry['compressed_bytes']) as response, temporary.open('wb') as target:
        while consumed < entry['compressed_bytes']:
            chunk = response.read(min(1024 * 1024, entry['compressed_bytes'] - consumed))
            if not chunk:
                raise RuntimeError('Truncated member stream')
            consumed += len(chunk)
            data = decoder.decompress(chunk)
            written += len(data)
            if written > entry['bytes']:
                raise RuntimeError('Member exceeds declared output size')
            crc = zlib.crc32(data, crc); digest.update(data); target.write(data)
            if consumed // (32 * 1024 * 1024) != (consumed - len(chunk)) // (32 * 1024 * 1024):
                print(json.dumps({'member': output.name, 'compressed_bytes_received': consumed}), flush=True)
        data = decoder.flush(); written += len(data); crc = zlib.crc32(data, crc); digest.update(data); target.write(data)
    if not decoder.eof or written != entry['bytes'] or crc != entry['crc32']:
        raise RuntimeError('ZIP size/CRC validation failed')
    temporary.replace(output)
    return {'filename': output.name, 'bytes': written, 'compressed_bytes_fetched': consumed,
            'crc32': f'{crc:08x}', 'crc32_verified': True, 'sha256': digest.hexdigest()}


def runtime_version(path):
    class ApiBase(ctypes.Structure):
        _fields_ = [('get_api', ctypes.c_void_p), ('get_version', ctypes.CFUNCTYPE(ctypes.c_char_p))]
    library = ctypes.CDLL(str(path), winmode=8)
    function = library.OrtGetApiBase; function.restype = ctypes.POINTER(ApiBase)
    return function().contents.get_version().decode('ascii')


def dump_dependencies(dumpbin, output, existing):
    queue = [output / name for name in TARGETS]; visited = {}; missing = set()
    system = Path('C:/Windows/System32')
    while queue:
        path = queue.pop(0)
        if path.name.lower() in visited:
            continue
        result = subprocess.run([str(dumpbin), '/DEPENDENTS', str(path)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                check=True, creationflags=0x08000000)
        text = result.stdout.decode('utf-8', errors='replace')
        (output / (path.name + '.dependents.txt')).write_text(text, 'utf-8')
        rows = []
        for name in re.findall(r'^\s+([a-zA-Z0-9_.-]+\.dll)\s*$', text, re.M | re.I):
            found = next((folder / name for folder in (output, existing, system) if folder and (folder / name).exists()), None)
            if name.lower().startswith(('api-ms-', 'ext-ms-')):
                location = 'Windows API set'
            elif found:
                location = str(found)
                if found.parent != system:
                    queue.append(found)
            else:
                location = None; missing.add(name)
            rows.append({'dll': name, 'resolved_readonly_path': location})
        visited[path.name.lower()] = {'path': str(path), 'dependencies': rows}
    return {'non_system_module_closure': visited, 'missing_dlls': sorted(missing)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu-dll', type=Path, required=True)
    parser.add_argument('--existing-cuda-dir', type=Path)
    parser.add_argument('--dumpbin', type=Path,
                        default=shutil.which('dumpbin.exe') or shutil.which('dumpbin'),
                        help='MSVC dumpbin; defaults to PATH in a Developer shell')
    args = parser.parse_args()
    if args.dumpbin is None:
        parser.error('dumpbin was not found on PATH; use an MSVC Developer shell or pass --dumpbin')
    cpu = args.cpu_dll.resolve(strict=True); version = runtime_version(cpu)
    if not version.startswith('1.24.'):
        raise RuntimeError('This experiment is restricted to the matching ORT 1.24.x ABI')
    metadata_url = f'https://pypi.org/pypi/onnxruntime-gpu/{version}/json'
    with urllib.request.urlopen(metadata_url, timeout=30) as response:
        metadata = json.load(response)
    wheel = next(item for item in metadata['urls'] if 'cp312-cp312-win_amd64.whl' in item['filename'])
    members = zip_members(wheel['url'], wheel['size'])
    selected = [members['onnxruntime/capi/' + name] for name in TARGETS]
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    receipt_file = output / 'preparation.json'
    if receipt_file.exists() or any((output / name).exists() for name in TARGETS):
        raise RuntimeError('Refusing to replace an existing isolated runtime')
    receipt = {'prepared': False, 'official_metadata_url': metadata_url, 'wheel_url': wheel['url'],
        'wheel_bytes': wheel['size'], 'wheel_sha256_published_not_full_download_verified': wheel['digests']['sha256'],
        'version': version, 'baseline_cpu_dll': str(cpu), 'baseline_cpu_sha256': hashlib.sha256(cpu.read_bytes()).hexdigest(),
        'selected_members': [], 'sessions_created': 0, 'models_loaded': 0,
        'system_install_or_formal_directory_changes': False, 'cudnn_downloaded': False}
    try:
        for entry in selected:
            item = extract_member(wheel['url'], entry, output / Path(entry['name']).name)
            receipt['selected_members'].append(item); print(json.dumps(item), flush=True)
        receipt['dependencies'] = dump_dependencies(args.dumpbin, output, args.existing_cuda_dir)
        receipt['prepared'] = True
    finally:
        receipt_file.write_text(json.dumps(receipt, indent=2), 'utf-8')
    print(json.dumps({'prepared': True, 'version': version, 'compressed_bytes_fetched': sum(item['compressed_bytes_fetched'] for item in receipt['selected_members']),
                      'missing_dlls': receipt['dependencies']['missing_dlls'], 'sessions_created': 0, 'models_loaded': 0}), flush=True)


if __name__ == '__main__':
    main()
