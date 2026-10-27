"""Static compatibility plus isolated no-model C ABI tests for the ASR cap."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "evidence/cuda-route/single-output"
DUMP = None


class ContextParams(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint32) for name in ["n_ctx", "n_batch", "n_ubatch", "n_seq_max"]]
    _fields_ += [(name, ctypes.c_int32) for name in ["n_threads", "n_threads_batch", "rope_scaling_type", "pooling_type", "attention_type", "flash_attn_type"]]
    _fields_ += [(name, ctypes.c_float) for name in ["rope_freq_base", "rope_freq_scale", "yarn_ext_factor", "yarn_attn_factor", "yarn_beta_fast", "yarn_beta_slow"]]
    _fields_ += [("yarn_orig_ctx", ctypes.c_uint32), ("defrag_thold", ctypes.c_float)]
    _fields_ += [("cb_eval", ctypes.c_void_p), ("cb_eval_user_data", ctypes.c_void_p)]
    _fields_ += [("type_k", ctypes.c_int32), ("type_v", ctypes.c_int32)]
    _fields_ += [("abort_callback", ctypes.c_void_p), ("abort_callback_data", ctypes.c_void_p)]
    _fields_ += [(name, ctypes.c_bool) for name in ["embeddings", "offload_kqv", "no_perf", "op_offload", "swa_full", "kv_unified"]]
    _fields_ += [("samplers", ctypes.c_void_p), ("n_samplers", ctypes.c_size_t)]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(*args):
    return subprocess.check_output([str(DUMP), *map(str, args)], creationflags=0x08000000).decode("utf-8", "replace").replace("\r\n", "\n")


def exports(path):
    data = dump("/exports", path)
    return data, set(re.findall(r"^\s+\d+\s+[0-9A-F]+\s+[0-9A-F]+\s+(\w+)\s*$", data, re.M))


def child(out):
    native_bin = out / "loader-bin"
    assert ctypes.sizeof(ContextParams) == 136
    directory = os.add_dll_directory(str(native_bin))
    libs = [ctypes.CDLL(str(native_bin / name), winmode=8) for name in ["ggml-base.dll", "ggml.dll", "llama.dll"]]
    native = libs[-1]
    default = native.llama_context_default_params
    default.argtypes, default.restype = [], ContextParams
    checks = {"parameter_size_is_136": ctypes.sizeof(ContextParams) == 136}
    for symbol in ["caps_llama_output_limit", "caps_llama_logits_width"]:
        function = getattr(native, symbol)
        function.argtypes, function.restype = [ctypes.c_void_p], ctypes.c_uint32
        checks[symbol + "_null_is_zero"] = function(None) == 0
    for symbol, limits in [("caps_llama_init_from_model_with_output_limit", [0, 1, 2, 0xFFFFFFFF]),
                           ("caps_llama_init_from_model_with_logits_limit", [0, 4000])]:
        function = getattr(native, symbol)
        function.argtypes, function.restype = [ctypes.c_void_p, ContextParams, ctypes.c_uint32], ctypes.c_void_p
        for limit in limits:
            checks[symbol + f"_null_model_limit_{limit}"] = not function(None, default(), limit)
    native.llama_init_from_model.argtypes, native.llama_init_from_model.restype = [ctypes.c_void_p, ContextParams], ctypes.c_void_p
    checks["original_init_null_model_rejected"] = not native.llama_init_from_model(None, default())
    assert all(checks.values())
    (out / "no-model-checks.json").write_text(json.dumps({"passed": True, "checks": checks,
        "models_loaded": 0, "backend_init_called": False, "candidate_sha256": digest(native_bin / "llama.dll")}, indent=2), "utf-8")
    directory.close()


def main():
    global DUMP
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-bin", type=Path)
    parser.add_argument("--candidate", type=Path, default=ROOT / ".build/asr-single-output/bin/llama.dll")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--source-dir", type=Path,
                        help="Prepared source directory; defaults to output-dir/native-src")
    parser.add_argument("--dumpbin", type=Path, default=shutil.which("dumpbin.exe"),
                        help="MSVC dumpbin executable; defaults to the developer shell PATH")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if args.child:
        return child(out)
    if args.original_bin is None or args.dumpbin is None:
        parser.error("--original-bin and --dumpbin (or dumpbin.exe on PATH) are required")
    DUMP = args.dumpbin.resolve(strict=True)
    built = args.candidate.resolve(strict=True)
    source_dir = (args.source_dir or out / "native-src").resolve(strict=True)
    original = args.original_bin.resolve(strict=True)
    native_bin = out / "loader-bin"
    native_bin.mkdir(parents=True, exist_ok=True)
    if built != out / "llama.dll":
        shutil.copy2(built, out / "llama.dll")
    imports = dump("/imports", built)
    (out / "candidate-imports.txt").write_text(imports, "utf-8")
    compatibility = []
    for name in ["ggml-base.dll", "ggml.dll"]:
        text, available = exports(original / name)
        (out / ("original-" + name + "-exports.txt")).write_text(text, "utf-8")
        parts = imports.split("\n    " + name + "\n", 1)
        assert len(parts) == 2, name
        section = re.split(r"^\s+[^\s]+\.dll\s*$", parts[1], maxsplit=1, flags=re.M)[0]
        required = set(re.findall(r"^\s+[0-9A-F]+\s+(ggml_\w+)\s*$", section, re.M))
        assert required and not required - available, (name, sorted(required - available))
        compatibility.append({"dll": name, "required_count": len(required), "missing": [],
                              "original_sha256": digest(original / name), "symbols": sorted(required)})
    old_text, old_exports = exports(original / "llama.dll")
    new_text, new_exports = exports(built)
    (out / "candidate-exports.txt").write_text(new_text, "utf-8")
    (out / "original-llama-exports.txt").write_text(old_text, "utf-8")
    assert not old_exports - new_exports
    assert {"caps_llama_init_from_model_with_output_limit", "caps_llama_output_limit"} <= new_exports
    receipt = json.loads((out / "source-receipt.json").read_text("utf-8"))
    for row in receipt["files"]:
        assert digest(source_dir / row["path"]) == row["candidate_sha256"], "Prepared source changed: " + row["path"]
    changed = [row["path"] for row in receipt["files"] if row["base_sha256"] != row["candidate_sha256"]]
    assert sorted(changed) == sorted(receipt["changed"])
    base_header = (Path(receipt["base_source"]) / "include/llama.h").read_text("utf-8")
    new_header = (source_dir / "include/llama.h").read_text("utf-8")
    pattern = r"    struct llama_context_params \{.*?\n    \};"
    assert re.search(pattern, base_header, re.S)[0] == re.search(pattern, new_header, re.S)[0]
    # The real logical guard precedes every mutation/inference step in decode.
    source = (source_dir / "src/llama-context.cpp").read_text("utf-8")
    decode = source.split("int llama_context::decode(", 1)[1].split("size_t llama_context::output_reserve", 1)[0]
    logical_guard = decode.index("n_outputs_all > max_output_rows")
    assert all(logical_guard < decode.index(anchor) for anchor in ["n_queued_tokens +=", "sched_reserve();", "memory_update(false)", "output_reserve(n_outputs_all)"])
    for name in ["ggml-base.dll", "ggml.dll", "llama.dll"]:
        shutil.copy2(built if name == "llama.dll" else original / name, native_bin / name)
    with (out / "no-model.stdout.txt").open("wb") as stdout, (out / "no-model.stderr.txt").open("wb") as stderr:
        run = subprocess.run([sys.executable, __file__, "--child", "--output-dir", str(out)], stdin=subprocess.DEVNULL,
                             stdout=stdout, stderr=stderr, timeout=30, creationflags=0x08000000)
    assert run.returncode == 0, "See preserved no-model stderr"
    report = {"passed": True, "candidate": str(out / "llama.dll"), "sha256": digest(built),
              "input_candidate": str(built), "source_dir": str(source_dir), "dumpbin": str(DUMP),
              "bytes": built.stat().st_size, "imports": compatibility, "public_parameter_abi_unchanged": True,
              "old_exports_preserved": len(old_exports), "added_exports": sorted(new_exports - old_exports),
              "only_three_source_files_changed_from_accepted_4k_base": True,
              "real_batch_guard_before_kv_and_compute": True, "models_loaded": 0,
              "not_yet_verified": ["real default-full control", "real single-row equivalence", "memory and speed", "GPU backend execution"]}
    (out / "static-compatibility.json").write_text(json.dumps(report, indent=2), "utf-8")
    print(json.dumps({"passed": True, "sha256": report["sha256"], "bytes": report["bytes"],
                      "imports": {row["dll"]: row["required_count"] for row in compatibility},
                      "models_loaded": 0}))


if __name__ == "__main__":
    main()
