"""Real-model API boundary probe; run serially under the benchmark scheduler.

No ASR encoder or sampler-generated expected result is used. The same external
prompt/token is computed through ordinary and capped contexts and raw full
vocabulary bytes are compared. The parent owns and times out one native child.
"""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from cuda_blas_smoke import bind, digest
from verify_asr_single_output import ContextParams


class ModelParams(C.Structure):
    _fields_ = [("devices", C.POINTER(C.c_void_p)), ("tensor_buft_overrides", C.c_void_p),
        ("n_gpu_layers", C.c_int32), ("split_mode", C.c_int32), ("main_gpu", C.c_int32),
        ("tensor_split", C.c_void_p), ("progress_callback", C.c_void_p),
        ("progress_callback_user_data", C.c_void_p), ("kv_overrides", C.c_void_p)]
    _fields_ += [(name, C.c_bool) for name in ["vocab_only", "use_mmap", "use_direct_io", "use_mlock",
        "check_tensors", "use_extra_bufts", "no_host", "no_alloc"]]


class Batch(C.Structure):
    _fields_ = [("n_tokens", C.c_int32), ("token", C.POINTER(C.c_int32)),
        ("embd", C.POINTER(C.c_float)), ("pos", C.POINTER(C.c_int32)),
        ("n_seq_id", C.POINTER(C.c_int32)), ("seq_id", C.POINTER(C.POINTER(C.c_int32))),
        ("logits", C.POINTER(C.c_int8))]


def save(path, report):
    path.write_text(json.dumps(report, indent=2), "utf-8")


def make_batch(payload, rows, flags, offset=0, token_ids=None, sequence=0):
    # Match production's three identical MRoPE planes plus one zero plane.
    position = (C.c_int32 * (rows * 4))(*([offset + i for i in range(rows)] * 3 + [0] * rows))
    n_seq = (C.c_int32 * rows)(*[1] * rows)
    zero = C.c_int32(sequence)
    seq = (C.POINTER(C.c_int32) * rows)(*[C.pointer(zero)] * rows)
    outputs = (C.c_int8 * rows)(*flags)
    token = (C.c_int32 * rows)(*token_ids) if token_ids is not None else None
    batch = Batch(rows, token, None if token is not None else payload, position, n_seq, seq, outputs)
    return batch, (position, n_seq, zero, seq, outputs, token, payload)


def child(args):
    assert (C.sizeof(ModelParams), C.sizeof(ContextParams), C.sizeof(Batch)) == (72, 136, 56)
    bind(C.WinDLL("kernel32"), "SetErrorMode", [C.c_uint32], C.c_uint32)(0x8003)
    native = args.app_dir.resolve() / "core/server/engines/llama/bin"
    raw = args.prefill_f32.read_bytes()
    if raw.startswith(b"\x93NUMPY") or not raw or len(raw) % 4096:
        raise ValueError("A complete raw little-endian [N,1024] prompt is required")
    n = len(raw) // 4096
    if not 513 <= n <= args.n_ctx - 2:
        raise ValueError("Need at least 513 prompt rows to test cross-microbatch rejection, with two spare KV slots")
    payload = (C.c_float * (n * 1024)).from_buffer_copy(raw)
    tokens = json.loads(args.tokens_json.read_text("utf-8"))
    if not isinstance(tokens, list) or not tokens or any(type(t) is not int for t in tokens):
        raise ValueError("External tokens must be a nonempty integer JSON array")
    report = {"schema": 1, "passed": False, "checks": {}, "n_prefill": n, "n_ubatch": 512,
        "backend": args.backend, "input_prefill_sha256": hashlib.sha256(raw).hexdigest(),
        "input_tokens_sha256": digest(args.tokens_json), "models_loaded": 0,
        "reference": "Same DLL's ordinary full context, same external prompt and token; no expected ASR tokens generated"}
    save(args.output, report)
    libraries = [C.CDLL(str(native / name), winmode=8) for name in ["ggml-base.dll", "ggml.dll", "libomp140.x86_64.dll", "llama.dll"]]
    ggml, lib = libraries[1], libraries[-1]
    load_backends = bind(ggml, "ggml_backend_load_all_from_path", [C.c_char_p], None)
    load_backends(str(native).encode("utf-8"))
    bind(lib, "llama_backend_init", [], None)()
    default_model = bind(lib, "llama_model_default_params", [], ModelParams)
    default_context = bind(lib, "llama_context_default_params", [], ContextParams)
    load_model = bind(lib, "llama_model_load_from_file", [C.c_char_p, ModelParams], C.c_void_p)
    free_model = bind(lib, "llama_model_free", [C.c_void_p], None)
    init_full = bind(lib, "llama_init_from_model", [C.c_void_p, ContextParams], C.c_void_p)
    init_one = bind(lib, "caps_llama_init_from_model_with_output_limit", [C.c_void_p, ContextParams, C.c_uint32], C.c_void_p)
    limit = bind(lib, "caps_llama_output_limit", [C.c_void_p], C.c_uint32)
    width = bind(lib, "caps_llama_logits_width", [C.c_void_p], C.c_uint32)
    free_context = bind(lib, "llama_free", [C.c_void_p], None)
    get_vocab = bind(lib, "llama_model_get_vocab", [C.c_void_p], C.c_void_p)
    vocab_count = bind(lib, "llama_vocab_n_tokens", [C.c_void_p], C.c_int32)
    model_dim = bind(lib, "llama_model_n_embd", [C.c_void_p], C.c_int32)
    decode = bind(lib, "llama_decode", [C.c_void_p, Batch], C.c_int32)
    get_memory = bind(lib, "llama_get_memory", [C.c_void_p], C.c_void_p)
    clear = bind(lib, "llama_memory_clear", [C.c_void_p, C.c_bool], None)
    max_pos = bind(lib, "llama_memory_seq_pos_max", [C.c_void_p, C.c_int32], C.c_int32)
    get_logits = bind(lib, "llama_get_logits_ith", [C.c_void_p, C.c_int32], C.POINTER(C.c_float))
    token_batch = bind(lib, "llama_batch_get_one", [C.POINTER(C.c_int32), C.c_int32], Batch)
    embedding_mode = bind(lib, "llama_set_embeddings", [C.c_void_p, C.c_bool], None)
    causal_mode = bind(lib, "llama_set_causal_attn", [C.c_void_p, C.c_bool], None)
    greedy_init = bind(lib, "llama_sampler_init_greedy", [], C.c_void_p)
    sampler_free = bind(lib, "llama_sampler_free", [C.c_void_p], None)
    set_sampler = bind(lib, "llama_set_sampler", [C.c_void_p, C.c_int32, C.c_void_p], C.c_bool)
    sample = bind(lib, "llama_sampler_sample", [C.c_void_p, C.c_void_p, C.c_int32], C.c_int32)
    params = default_model()
    params.n_gpu_layers = -1
    selected = (C.c_void_p * 2)()
    if args.backend != "cpu":
        name = {"cuda": b"CUDA0", "vulkan": b"Vulkan0"}[args.backend]
        selected[0] = bind(ggml, "ggml_backend_dev_by_name", [C.c_char_p], C.c_void_p)(name)
        if not selected[0]:
            raise RuntimeError("Requested GPU device not available")
    params.devices = selected
    model = context = sampler = None
    try:
        weights = args.app_dir.resolve() / "models/Qwen3-ASR/Qwen3-ASR-0.6B/qwen3_asr_llm.q4_k.gguf"
        report["weights_sha256"] = digest(weights)
        report["native_llama_sha256"] = digest(native / "llama.dll")
        model = load_model(str(weights).encode("utf-8"), params)
        if not model:
            raise RuntimeError("Model loading failed; inspect native stderr")
        report["models_loaded"] = 1
        vocab = vocab_count(get_vocab(model))
        if model_dim(model) != 1024 or not all(0 <= t < vocab for t in tokens):
            raise ValueError("Input dimension or token vocabulary mismatch")
        settings = default_context()
        settings.n_ctx, settings.n_batch, settings.n_ubatch, settings.n_seq_max = args.n_ctx, 4096, 512, 1
        settings.n_threads = settings.n_threads_batch = 1
        settings.embeddings, settings.pooling_type, settings.flash_attn_type = False, 0, 1
        settings.no_perf, settings.offload_kqv = True, True

        def checked(name, condition):
            report["checks"][name] = bool(condition)
            save(args.output, report)
            if not condition:
                raise AssertionError(name)

        def snapshot(ctx):
            pointer = get_logits(ctx, -1)  # synchronizes the native backend
            if not pointer:
                raise RuntimeError("Last full-vocabulary logits pointer is null")
            values = [pointer[i] for i in range(vocab)]
            if not all(__import__("math").isfinite(value) for value in values):
                raise RuntimeError("Nonfinite full-vocabulary logits")
            return hashlib.sha256(C.string_at(pointer, vocab * 4)).hexdigest(), max(range(vocab), key=values.__getitem__)

        def prefill(ctx, positions=None):
            flags = [0] * n
            for i in (positions if positions is not None else [n - 1]):
                flags[i] = 1
            batch, keep = make_batch(payload, n, flags)
            return decode(ctx, batch)

        # A valid model lets these tests reach actual argument validation.
        for row_limit in [0, 2, 0xFFFFFFFF]:
            invalid = init_one(model, settings, row_limit)
            if invalid:
                free_context(invalid)
            checked(f"invalid_limit_{row_limit}_rejected", not invalid)
        for name, field, value in [("embedding", "embeddings", True), ("multisequence", "n_seq_max", 2),
                                  ("noncausal", "attention_type", 1), ("pooling", "pooling_type", 1),
                                  ("backend_sampler_count", "n_samplers", 1)]:
            modified = ContextParams.from_buffer_copy(settings)
            setattr(modified, field, value)
            invalid = init_one(model, modified, 1)
            if invalid:
                free_context(invalid)
            checked(name + "_init_rejected", not invalid)

        context = init_full(model, settings)
        checked("ordinary_init_still_unlimited_full_vocab", bool(context) and limit(context) == 0 and width(context) == vocab)
        memory = get_memory(context)
        clear(memory, True)
        checked("ordinary_prefill_one_row", prefill(context) == 0)
        reference_prompt, reference_greedy = snapshot(context)
        external_token = C.c_int32(tokens[0])
        checked("ordinary_decode_one_token", decode(context, token_batch(C.byref(external_token), 1)) == 0)
        reference_next, _ = snapshot(context)
        clear(memory, False)
        checked("ordinary_two_rows_remain_supported", prefill(context, [0, n - 1]) == 0)
        checked("ordinary_two_rows_advance_kv", max_pos(memory, 0) == n - 1)
        free_context(context)
        context = None

        context = init_one(model, settings, 1)
        checked("private_init_one_row_full_vocab", bool(context) and limit(context) == 1 and width(context) == vocab)
        memory = get_memory(context)
        clear(memory, True)
        checked("cross_ubatch_two_rows_rejected", prefill(context, [0, n - 1]) == -1)
        checked("rejected_empty_kv_stays_empty", max_pos(memory, 0) == -1)
        checked("valid_prefill_after_rejection", prefill(context) == 0)
        observed_prompt, _ = snapshot(context)
        checked("full_vocab_prefill_exact_to_ordinary", observed_prompt == reference_prompt)
        invalid, keep = make_batch(None, 2, [1, 1], n, [tokens[0], tokens[0]])
        checked("two_token_rows_rejected_after_prefill", decode(context, invalid) == -1)
        checked("rejection_preserves_existing_kv", max_pos(memory, 0) == n - 1)
        checked("rejection_preserves_last_logits", snapshot(context)[0] == reference_prompt)
        invalid_sequence, keep_seq = make_batch(None, 1, [1], n, [tokens[0]], sequence=1)
        checked("sequence_one_rejected", decode(context, invalid_sequence) == -1)
        checked("sequence_rejection_preserves_kv", max_pos(memory, 0) == n - 1)
        sampler = greedy_init()
        checked("cpu_sampler_created", bool(sampler))
        checked("backend_sampler_switch_rejected", not set_sampler(context, 0, sampler))
        checked("cpu_full_vocab_sampler_still_supported", sample(sampler, context, -1) == reference_greedy)
        embedding_mode(context, True)
        causal_mode(context, False)
        # graph_reserve is an internal non-exported API in this Windows DLL.
        # Its shared clamp is checked by source audit and context recreation,
        # without introducing test-only native exports.
        checked("valid_decode_after_rejected_mode_switches", decode(context, token_batch(C.byref(external_token), 1)) == 0)
        checked("full_vocab_decode_exact_to_ordinary", snapshot(context)[0] == reference_next)
        clear(memory, False)
        checked("clear_then_prefill_recovers", prefill(context) == 0 and snapshot(context)[0] == reference_prompt)
        free_context(context)
        context = None
        context = init_one(model, settings, 1)
        checked("private_context_recreation", bool(context) and limit(context) == 1)
        clear(get_memory(context), True)
        checked("recreated_context_prefill_exact", prefill(context) == 0 and snapshot(context)[0] == reference_prompt)
        report["reference_prompt_logits_sha256"] = reference_prompt
        report["reference_next_logits_sha256"] = reference_next
        report["compared_logits_width"] = vocab
        report["passed"] = True
    finally:
        if sampler:
            sampler_free(sampler)
        if context:
            free_context(context)
        if model:
            free_model(model)
        save(args.output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=["cuda", "vulkan", "cpu"], default="cuda")
    parser.add_argument("--prefill-f32", type=Path, required=True)
    parser.add_argument("--tokens-json", type=Path, required=True)
    parser.add_argument("--n-ctx", type=int, choices=[2048, 4096], default=4096)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        return child(args)
    if not 1 <= args.timeout <= 600:
        parser.error("Timeout must be 1 to 600 seconds")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump({"passed": False, "stage": "owned_child_not_completed"}, stream)
    env = dict(os.environ)
    env["PATH"] = str(args.app_dir.resolve() / "core/server/engines/llama/bin") + os.pathsep + env.get("PATH", "")
    timed_out = False
    stderr_path = args.output.with_suffix(".stderr.txt")
    with args.output.with_suffix(".stdout.txt").open("xb") as stdout, stderr_path.open("xb") as stderr:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--child"],
            stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, env=env, creationflags=0x08000000)
        try:
            process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            process.wait(timeout=10)
    report = json.loads(args.output.read_text("utf-8"))
    report["launcher"] = {"pid": process.pid, "exit_code": process.returncode, "timed_out": timed_out, "waited": True}
    log = stderr_path.read_text("utf-8", errors="replace")
    if args.backend != "cpu":
        device = {"cuda": "CUDA0", "vulkan": "Vulkan0"}[args.backend]
        assignments = re.findall(r"load_tensors: layer\s+\d+ assigned to device ([^,\s]+)", log)
        offload = re.search(r"offloaded (\d+)/(\d+) layers to GPU", log)
        report["requested_gpu_verified"] = bool(assignments and all(name == device for name in assignments)
            and offload and int(offload[1]) > 0 and offload[1] == offload[2])
    report["passed"] = bool(report.get("passed") and process.returncode == 0 and not timed_out
                            and (args.backend == "cpu" or report.get("requested_gpu_verified")))
    save(args.output, report)
    if not report["passed"]:
        raise SystemExit("ASR output guard probe failed; inspect preserved JSON and stderr")
    print(json.dumps({"passed": True, "checks": len(report["checks"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
