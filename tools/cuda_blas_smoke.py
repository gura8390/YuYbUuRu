"""Exercise delayed cuBLAS loading with a model-free, known F32 CUDA matmul.

The parent launches one owned child, with exactly the worker's private PATH
prefix strategy. No AddDllDirectory/vendor preload, ORT, model or ASR is used.
The 64-column workload bypasses b9090 MMVF/MMF and reaches cuBLAS SGEMM.
"""
import argparse
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

SYMBOLS = ["cublasCreate_v2", "cublasDestroy_v2", "cublasGemmBatchedEx", "cublasGemmEx",
           "cublasGemmStridedBatchedEx", "cublasGetStatusString", "cublasSetMathMode",
           "cublasSetStream_v2", "cublasSgemmStridedBatched", "cublasSgemm_v2", "cublasStrsmBatched"]


def save(path, report):
    path.write_text(json.dumps(report, indent=2), "utf-8")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bind(lib, name, args, result):
    function = getattr(lib, name)
    function.argtypes, function.restype = args, result
    return function


def modules():
    kernel = C.WinDLL("kernel32", use_last_error=True)
    psapi = C.WinDLL("psapi", use_last_error=True)
    process = bind(kernel, "GetCurrentProcess", [], C.c_void_p)()
    enumerate_modules = bind(psapi, "EnumProcessModulesEx",
        [C.c_void_p, C.POINTER(C.c_void_p), C.c_uint32, C.POINTER(C.c_uint32), C.c_uint32], C.c_int)
    name = bind(psapi, "GetModuleFileNameExW", [C.c_void_p, C.c_void_p, C.c_wchar_p, C.c_uint32], C.c_uint32)
    capacity = 256
    while True:
        array = (C.c_void_p * capacity)()
        needed = C.c_uint32()
        if not enumerate_modules(process, array, C.sizeof(array), C.byref(needed), 3):
            raise C.WinError(C.get_last_error())
        if needed.value <= C.sizeof(array):
            break
        capacity = needed.value // C.sizeof(C.c_void_p) + 32
    result = []
    for handle in array[:needed.value // C.sizeof(C.c_void_p)]:
        buffer = C.create_unicode_buffer(32768)
        count = name(process, handle, buffer, len(buffer))
        if not count or count >= len(buffer):
            raise C.WinError(C.get_last_error())
        path = Path(buffer.value)
        if any(term in path.name.lower() for term in ["cublas", "cudart", "nvcuda", "ggml", "llama", "onnx"]):
            result.append({"name": path.name, "path": str(path)})
    return sorted(result, key=lambda item: item["name"].lower())


class InitParams(C.Structure):
    _fields_ = [("mem_size", C.c_size_t), ("mem_buffer", C.c_void_p), ("no_alloc", C.c_bool)]


def child(args):
    bind(C.WinDLL("kernel32"), "SetErrorMode", [C.c_uint32], C.c_uint32)(0x8003)
    native = args.native_bin.resolve(strict=True)
    report = {"schema": 1, "passed": False, "models_loaded": 0, "ort_loaded": False,
              "native_bin": str(native), "pid": os.getpid(), "stage": "before_library_load",
              "matmul": {"shape": [64, 64], "type": "F32", "formula": "A transpose times B with A=identity",
                          "B": "((column*17+row)%129-64)/16; exactly representable in F32 and TF32"},
              "cuBLAS_boundary": "Known SGEMM result plus resolution of all 11 original imports; other cuBLAS operations are not numerically tested"}
    save(args.output, report)
    backend = context = buffer = None
    api = {}
    try:
        base = C.CDLL(str(native / "ggml-base.dll"), winmode=8)
        cuda = C.CDLL(str(native / "ggml-cuda.dll"), winmode=8)
        report["dll_sha256"] = {name: digest(native / name) for name in ["ggml-base.dll", "ggml-cuda.dll"]}
        if args.expected_plugin_sha256 and report["dll_sha256"]["ggml-cuda.dll"] != args.expected_plugin_sha256.lower():
            raise RuntimeError("Plugin hash differs from the requested candidate")
        api["init"] = bind(base, "ggml_init", [InitParams], C.c_void_p)
        api["free"] = bind(base, "ggml_free", [C.c_void_p], None)
        api["tensor"] = bind(base, "ggml_new_tensor_2d", [C.c_void_p, C.c_int, C.c_int64, C.c_int64], C.c_void_p)
        api["mul_mat"] = bind(base, "ggml_mul_mat", [C.c_void_p, C.c_void_p, C.c_void_p], C.c_void_p)
        api["graph"] = bind(base, "ggml_new_graph", [C.c_void_p], C.c_void_p)
        api["expand"] = bind(base, "ggml_build_forward_expand", [C.c_void_p, C.c_void_p], None)
        api["nbytes"] = bind(base, "ggml_nbytes", [C.c_void_p], C.c_size_t)
        api["alloc"] = bind(base, "ggml_backend_alloc_ctx_tensors", [C.c_void_p, C.c_void_p], C.c_void_p)
        api["buffer_free"] = bind(base, "ggml_backend_buffer_free", [C.c_void_p], None)
        api["backend_free"] = bind(base, "ggml_backend_free", [C.c_void_p], None)
        api["backend_name"] = bind(base, "ggml_backend_name", [C.c_void_p], C.c_char_p)
        api["supports"] = bind(base, "ggml_backend_supports_op", [C.c_void_p, C.c_void_p], C.c_bool)
        api["set"] = bind(base, "ggml_backend_tensor_set", [C.c_void_p, C.c_void_p, C.c_size_t, C.c_size_t], None)
        api["get"] = bind(base, "ggml_backend_tensor_get", [C.c_void_p, C.c_void_p, C.c_size_t, C.c_size_t], None)
        api["compute"] = bind(base, "ggml_backend_graph_compute", [C.c_void_p, C.c_void_p], C.c_int)
        api["sync"] = bind(base, "ggml_backend_synchronize", [C.c_void_p], None)
        cuda_init = bind(cuda, "ggml_backend_cuda_init", [C.c_int], C.c_void_p)
        report["modules_after_plugin_load"] = modules()
        if any("cublas" in row["name"].lower() for row in report["modules_after_plugin_load"]):
            raise RuntimeError("cuBLAS was already loaded: this does not prove a delayed-load candidate")
        backend = cuda_init(0)
        if not backend:
            raise RuntimeError("CUDA0 backend initialization failed")
        report["backend"] = api["backend_name"](backend).decode("utf-8", "replace")
        if report["backend"] != "CUDA0":
            raise RuntimeError("Unexpected backend; CPU fallback is not accepted")
        assert C.sizeof(InitParams) == 24
        context = api["init"](InitParams(8 * 1024 * 1024, None, True))
        if not context:
            raise RuntimeError("ggml metadata context initialization failed")
        a = api["tensor"](context, 0, 64, 64)
        b = api["tensor"](context, 0, 64, 64)
        product = api["mul_mat"](context, a, b)
        graph = api["graph"](context)
        if not all([a, b, product, graph]) or api["nbytes"](product) != 64 * 64 * 4:
            raise RuntimeError("Unexpected graph allocation or output shape")
        if not api["supports"](backend, product):
            raise RuntimeError("CUDA backend does not support the F32 matmul")
        api["expand"](graph, product)
        buffer = api["alloc"](context, backend)
        if not buffer:
            raise RuntimeError("CUDA tensor buffer allocation failed")
        identity = (C.c_float * 4096)(*[float(row == column) for column in range(64) for row in range(64)])
        expected = [((column * 17 + row) % 129 - 64) / 16 for column in range(64) for row in range(64)]
        values = (C.c_float * 4096)(*expected)
        output = (C.c_float * 4096)()
        api["set"](a, identity, 0, C.sizeof(identity))
        api["set"](b, values, 0, C.sizeof(values))
        report["modules_before_compute"] = modules()
        if any("cublas" in row["name"].lower() for row in report["modules_before_compute"]):
            raise RuntimeError("cuBLAS loaded before graph computation; delayed SGEMM proof is ambiguous")
        report["stage"] = "before_cublas_graph_compute"
        save(args.output, report)
        started = time.perf_counter()
        status = api["compute"](backend, graph)
        api["sync"](backend)
        report["compute_synchronized_seconds"] = time.perf_counter() - started
        report["graph_status"] = status
        if status != 0:
            raise RuntimeError(f"CUDA graph returned status {status}")
        api["get"](product, output, 0, C.sizeof(output))
        report["modules_after_compute"] = modules()
        by_name = {row["name"].lower(): row for row in report["modules_after_compute"]}
        for name in ["cublas64_12.dll", "cublaslt64_12.dll"]:
            if name not in by_name or Path(by_name[name]["path"]).resolve() != (native / name).resolve():
                raise RuntimeError(f"Expected private vendor DLL did not load from the native directory: {name}")
        if any("onnx" in name or "llama" in name for name in by_name):
            raise RuntimeError("Unexpected model/ORT library was loaded")
        kernel = C.WinDLL("kernel32", use_last_error=True)
        handle = bind(kernel, "GetModuleHandleW", [C.c_wchar_p], C.c_void_p)("cublas64_12.dll")
        address = bind(kernel, "GetProcAddress", [C.c_void_p, C.c_char_p], C.c_void_p)
        report["original_11_cublas_symbols_resolved"] = {name: bool(address(handle, name.encode("ascii"))) for name in SYMBOLS}
        if not all(report["original_11_cublas_symbols_resolved"].values()):
            raise RuntimeError("An original cuBLAS import is absent from the actual loaded DLL")
        observed = list(output)
        differences = [abs(x - y) for x, y in zip(observed, expected)]
        report["numeric"] = {"all_finite": all(math.isfinite(x) for x in observed),
                             "max_absolute_error": max(differences), "count": len(observed),
                             "expected_sha256": hashlib.sha256(struct.pack("<4096f", *expected)).hexdigest(),
                             "output_sha256": hashlib.sha256(bytes(output)).hexdigest()}
        if not report["numeric"]["all_finite"] or max(differences) != 0:
            raise RuntimeError("Known identity matmul result is not exact")
        report["passed"] = True
        report["stage"] = "complete"
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        cleanup_errors = []
        for action, handle in [("sync", backend), ("buffer_free", buffer), ("free", context), ("backend_free", backend)]:
            if handle and action in api:
                try:
                    api[action](handle)
                except BaseException as error:
                    cleanup_errors.append(f"{action}: {error}")
        report["cleanup_errors"] = cleanup_errors
        if cleanup_errors:
            report["passed"] = False
        save(args.output, report)
    if not report["passed"]:
        raise RuntimeError("Smoke validation/cleanup failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-bin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--expected-plugin-sha256")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if sys.platform != "win32":
        parser.error("This diagnostic requires Windows x64")
    if args.child:
        return child(args)
    if not 1 <= args.timeout <= 300:
        parser.error("Timeout must be between 1 and 300 seconds")
    native = args.native_bin.resolve(strict=True)
    for name in ["ggml-base.dll", "ggml-cuda.dll", "cublas64_12.dll", "cublasLt64_12.dll"]:
        if not (native / name).is_file():
            raise FileNotFoundError(native / name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump({"passed": False, "stage": "child_not_completed", "models_loaded": 0}, stream)
    env = dict(os.environ)
    env["PATH"] = str(native) + os.pathsep + env.get("PATH", "")
    stdout_path = args.output.with_suffix(".stdout.txt")
    stderr_path = args.output.with_suffix(".stderr.txt")
    timed_out = False
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--child"],
            stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, env=env, creationflags=0x08000000)
        try:
            process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            process.wait(timeout=10)
    report = json.loads(args.output.read_text("utf-8"))
    report["launcher"] = {"owned_child_pid": process.pid, "exit_code": process.returncode,
        "timed_out": timed_out, "timeout_seconds": args.timeout, "private_child_PATH_prefix": str(native),
        "stdout": str(stdout_path), "stderr": str(stderr_path), "cleanup": "owned child waited; timeout kills only this PID"}
    if process.returncode or timed_out:
        report["passed"] = False
    save(args.output, report)
    if not report["passed"]:
        raise SystemExit("CUDA cuBLAS smoke failed; inspect preserved JSON and native stderr")
    print(json.dumps({"passed": True, "output": str(args.output), "max_absolute_error": report["numeric"]["max_absolute_error"]}))


if __name__ == "__main__":
    main()
