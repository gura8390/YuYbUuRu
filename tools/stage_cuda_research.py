"""Create isolated CUDA/Vulkan research runtimes; never alter an installed app.

Models, assets and the CPU ORT directory are shared junctions, logically read-only
by convention only: no OS write protection is applied. Do not modify their data.
Native binaries remain shared immutable links/copies; mutable hot*.txt files are
independent copies. An existing destination is refused. The baseline DLL hash
defaults to the accepted 0.4.3 build, not an arbitrary compatible user library.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def link_file(source, target):
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)
    assert sha(source) == sha(target)


def junction(target, source):
    subprocess.run(['cmd.exe', '/d', '/c', 'mklink', '/J', str(target), str(source)],
                   check=True, capture_output=True, creationflags=0x08000000)
    assert target.resolve() == source.resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-app', required=True, type=Path)
    parser.add_argument('--cuda-native-dir', required=True, type=Path)
    parser.add_argument('--delay-plugin', required=True, type=Path)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'evidence/runtimes')
    parser.add_argument('--expected-llama-sha256',
                        default='38a775283f6396d83cc7d584e49f06f9e8d4a1ad1dae1d0fee8e3a9b644cb070',
                        help='Accepted baseline DLL SHA256; defaults to the fixed 0.4.3 build')
    args = parser.parse_args()
    expected_llama = args.expected_llama_sha256.lower()
    if len(expected_llama) != 64 or any(c not in '0123456789abcdef' for c in expected_llama):
        parser.error('--expected-llama-sha256 must be exactly 64 hexadecimal characters')
    app = args.baseline_app.resolve(strict=True)
    cuda = args.cuda_native_dir.resolve(strict=True)
    delay = args.delay_plugin.resolve(strict=True)
    out = args.output_root.resolve()
    assert out.is_relative_to(ROOT.resolve()) and not out.exists()
    native = Path('core/server/engines/llama/bin')
    if sha(app / native / 'llama.dll') != expected_llama:
        raise RuntimeError('Baseline llama.dll differs from the explicitly accepted SHA256')
    settings = json.loads((app / 'settings.json').read_text('utf-8'))
    results = {}
    for name, backend in [('vulkan', 'vulkan'), ('cuda-auto', 'cuda'), ('cuda-delay', 'cuda')]:
        target = out / name
        (target / native).mkdir(parents=True)
        for source in (app / native).glob('*.dll'):
            if source.name != 'ggml-vulkan.dll' or backend == 'vulkan':
                link_file(source, target / native / source.name)
        if backend == 'cuda':
            for filename in ['ggml-cuda.dll', 'cudart64_12.dll', 'cublas64_12.dll', 'cublasLt64_12.dll']:
                source = delay if name == 'cuda-delay' and filename == 'ggml-cuda.dll' else cuda / filename
                link_file(source, target / native / filename)
        for filename in ['models', 'internal', 'assets']:
            junction(target / filename, app / filename)
        for source in app.iterdir():
            if not source.is_file():
                continue
            if source.name.lower().startswith('hot') and source.suffix.lower() == '.txt':
                destination = target / source.name
                shutil.copy2(source, destination)
                if sha(source) != sha(destination):
                    raise RuntimeError('Copied hotword file differs from baseline')
            elif source.suffix.lower() == '.dll' or source.name in ['ffmpeg.exe', 'ffprobe.exe']:
                link_file(source, target / source.name)
        policy = {**settings, 'gpu_backend': backend, 'vulkan_pipeline_cache': backend == 'vulkan'}
        (target / 'settings.json').write_text(json.dumps(policy, indent=2) + '\n', 'utf-8')
        results[name] = {'app_dir': str(target), 'backend': backend,
                         'native_files': {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size}
                                          for p in sorted((target / native).glob('*.dll'))},
                         'read_only_junctions': {key: str((target / key).resolve()) for key in ['models', 'internal', 'assets']},
                         'junction_os_write_protection': False, 'hotword_files_copied': True}
    receipt = {'passed': True, 'formal_install_modified': False,
               'expected_baseline_llama_sha256': expected_llama, 'variants': results}
    (out / 'staging.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), 'utf-8')
    print(json.dumps({'passed': True, 'variants': list(results), 'output': str(out)}))


if __name__ == '__main__':
    main()
