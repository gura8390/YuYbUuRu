"""Stage a GPU ORT candidate beside the CPU junction in an isolated research app.

No settings are changed and no DLL is loaded. The printed config fragment starts
with the GPU core's CPU EP control; CUDA provider experiments are a later step.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil

ROOT = Path(__file__).resolve().parents[1]
ORT = ('onnxruntime.dll', 'onnxruntime_providers_shared.dll', 'onnxruntime_providers_cuda.dll')
DEPS = ('cufft64_11.dll', 'cudnn64_9.dll', 'cudnn_graph64_9.dll', 'cudnn_ops64_9.dll', 'cudnn_cnn64_9.dll')
CUDA = ('cublas64_12.dll', 'cublasLt64_12.dll', 'cudart64_12.dll')
ENGINES = ('cudnn_engines_precompiled64_9.dll', 'cudnn_engines_runtime_compiled64_9.dll', 'cudnn_heuristic64_9.dll')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--app', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument('--core-only', action='store_true', help='Stage the three ORT DLLs for the GPU-core/CPU-EP control only')
    phase.add_argument('--complete-existing', action='store_true', help='Add verified dependencies to a previously staged core-only candidate')
    phase.add_argument('--add-frontend-engines', action='store_true', help='Add only the separately approved verified engines to an existing full candidate')
    args = parser.parse_args()
    app = args.app.resolve(strict=True)
    # Only research runtimes in this CUDA checkout are valid mutation targets.
    research = (ROOT / 'evidence/runtimes').resolve()
    if not app.is_relative_to(research) or app == research:
        raise RuntimeError('Target must be an existing research app beneath this checkout evidence/runtimes')
    prepared = args.prepared.resolve(strict=True)
    ort_receipt = json.loads((prepared / 'preparation.json').read_text('utf-8'))
    deps_receipt = json.loads((prepared / 'dependency-preparation.json').read_text('utf-8')) if not args.core_only else None
    if not ort_receipt.get('prepared') or (deps_receipt is not None and not deps_receipt.get('prepared')):
        raise RuntimeError('Required selective-download receipts must be complete')
    records = {row['filename']: row for row in ort_receipt['selected_members']}
    if deps_receipt is not None:
        records.update({row['filename']: row for row in deps_receipt['cudnn']['selected_members']})
        records.update({row['filename']: row for row in deps_receipt['cufft']['selected_members']})
    if args.add_frontend_engines:
        frontend = json.loads((prepared / 'frontend-engine-preparation.json').read_text('utf-8'))
        if not frontend.get('prepared') or frontend.get('version') != '9.7.1.26':
            raise RuntimeError('Frontend engines must be separately verified first')
        records.update({row['filename']: row for row in frontend['selected_members']})
    sources = {name: prepared / name for name in (ORT if args.core_only else (*ORT, *DEPS))}
    if args.add_frontend_engines:
        sources.update({name: prepared / name for name in ENGINES})
    for name, source in sources.items():
        record = records[name]
        if not record.get('crc32_verified') or sha(source) != record['sha256']:
            raise RuntimeError(f'Download receipt mismatch: {name}')
    native = app / 'core/server/engines/llama/bin'
    if not args.core_only:
        sources.update({name: native / name for name in CUDA})
    for source in sources.values():
        if not source.is_file():
            raise FileNotFoundError(source)
    target = app / 'runtime-ort-gpu/capi'
    if not target.resolve().is_relative_to(app):
        raise RuntimeError('Refusing existing or redirected candidate runtime')
    internal_before = (app / 'internal').resolve(strict=True)
    prior = None
    if args.complete_existing or args.add_frontend_engines:
        prior = json.loads((target / 'staging.json').read_text('utf-8'))
        expected = set(ORT) if args.complete_existing else set((*ORT, *DEPS, *CUDA))
        if (prior.get('core_only') != args.complete_existing or
                prior.get('internal_target') != str(internal_before)):
            raise RuntimeError('Existing runtime has an unexpected phase or CPU junction')
        if set(prior['files']) != expected or {path.name for path in target.iterdir()} != expected | {'staging.json'}:
            raise RuntimeError('Existing staged directory has unexpected files')
        for name in expected:
            if sha(target / name) != prior['files'][name]['sha256']:
                raise RuntimeError('Staged file changed')
            if name in records and sha(target / name) != records[name]['sha256']:
                raise RuntimeError('Staged file no longer matches download receipt')
    else:
        if target.exists():
            raise RuntimeError('Refusing to replace existing candidate runtime')
        target.mkdir(parents=True)
    files = {}
    for name, source in sources.items():
        destination = target / name
        if prior is not None and name in prior['files']:
            files[name] = prior['files'][name]
            continue
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
        digest = sha(destination)
        if digest != sha(source):
            raise RuntimeError(f'Staged file differs: {name}')
        files[name] = {'bytes': destination.stat().st_size, 'sha256': digest, 'source': str(source)}
    if (app / 'internal').resolve(strict=True) != internal_before:
        raise RuntimeError('CPU internal junction changed during staging')
    receipt = {'staged': True, 'core_only': args.core_only, 'frontend_engines': args.add_frontend_engines,
               'sessions_created': 0, 'models_loaded': 0, 'gpu_initialized': False,
               'internal_junction_unchanged': True, 'internal_target': str(internal_before),
               'candidate_dir': str(target), 'files': files,
               'initial_control_config': {'onnx_dll': 'runtime-ort-gpu/capi/onnxruntime.dll',
                                          'encoder_provider': 'cpu', 'encoder_cuda_mem_limit_mb': 512},
               'not_staged': ['cudnn_adv64_9.dll', *(() if args.add_frontend_engines else ENGINES)],
               'boundary': 'Static staging only; lazy cuDNN closure and provider placement require separate session validation'}
    (target / 'staging.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', 'utf-8')
    print(json.dumps({'staged': True, 'core_only': args.core_only, 'candidate_dir': str(target),
                      'initial_control_config': receipt['initial_control_config']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
