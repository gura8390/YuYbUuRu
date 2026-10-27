"""Static PE import and embedded cuDNN lazy-library evidence; never load a DLL."""
from pathlib import Path
import argparse
import hashlib
import json
import mmap
import re
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--existing-cuda-dir', type=Path, required=True)
    parser.add_argument('--dumpbin', type=Path,
                        default=shutil.which('dumpbin.exe') or shutil.which('dumpbin'),
                        help='MSVC dumpbin; defaults to PATH in a Developer shell')
    args = parser.parse_args()
    if args.dumpbin is None:
        parser.error('dumpbin was not found on PATH; use an MSVC Developer shell or pass --dumpbin')
    stage = args.prepared.resolve(strict=True)
    native = args.existing_cuda_dir.resolve(strict=True)
    system = Path('C:/Windows/System32')
    rows = {}
    missing, lazy_missing = set(), set()
    for dll in sorted(stage.glob('*.dll')):
        result = subprocess.run([str(args.dumpbin), '/DEPENDENTS', str(dll)], check=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, creationflags=0x08000000)
        text = result.stdout.decode('utf-8', errors='replace')
        (stage / (dll.name + '.dependents.txt')).write_text(text, 'utf-8')
        imports = []
        for name in re.findall(r'^\s+([a-zA-Z0-9_.-]+\.dll)\s*$', text, re.M | re.I):
            located = next((directory / name for directory in (stage, native, system)
                            if (directory / name).is_file()), None)
            api_set = name.lower().startswith(('api-ms-', 'ext-ms-'))
            if not located and not api_set:
                missing.add(name)
            imports.append({'name': name, 'resolved': str(located) if located else
                            ('Windows API set' if api_set else None)})
        lazy = []
        if dll.name.startswith('cudnn'):
            result = subprocess.run([str(args.dumpbin), '/IMPORTS', str(dll)], check=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, creationflags=0x08000000)
            (stage / (dll.name + '.imports.txt')).write_text(result.stdout.decode('utf-8', errors='replace'), 'utf-8')
            with dll.open('rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
                names = sorted({match.group().decode('ascii') for match in
                                re.finditer(rb'cudnn_[a-z0-9_]+64_9\.dll', data)})
            for name in names:
                exists = any((directory / name).is_file() for directory in (stage, native, system))
                lazy.append({'embedded_name': name, 'available_in_explicit_search_dirs': exists,
                             'required_by_this_model': 'unverified; an embedded reference is not a load attempt'})
                if not exists:
                    lazy_missing.add(name)
        with dll.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        rows[dll.name] = {'bytes': dll.stat().st_size, 'sha256': digest,
                          'pe_imports': imports, 'embedded_lazy_dll_names': lazy}
    report = {'gpu_initialized': False, 'sessions_created': 0, 'models_loaded': 0,
              'all_staged_pe_imports_present': not missing, 'missing_pe_imports': sorted(missing),
              'unavailable_embedded_lazy_library_names': sorted(lazy_missing), 'modules': rows,
              'boundary': 'PE imports are static evidence only. Lazy DLL requirements, SM120 execution, provider placement and numerical quality remain untested.'}
    path = stage / 'static-closure.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', 'utf-8')
    print(json.dumps({'static_pe_complete': not missing, 'missing_pe_imports': sorted(missing),
                      'unavailable_lazy_names': sorted(lazy_missing), 'report': str(path)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
