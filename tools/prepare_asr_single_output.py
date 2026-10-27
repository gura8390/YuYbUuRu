"""Copy the accepted limited-logits b9090 source and add an opt-in ASR row cap.

Only the isolated copy is patched. Public parameter structures, model weights,
normal initializers, token batches and the CPU alignment extension stay intact.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "evidence/cuda-route/single-output"
FILES = ["include/llama.h", "src/llama-context.h", "src/llama-context.cpp"]
# These are the accepted b9090 sources after aligner-logits.patch, measured
# from the existing build's base-source receipt. read_text normalizes CRLF/LF;
# hash UTF-8 text so Git's Windows line-ending conversion remains acceptable.
BASE_TEXT_SHA256 = {
    "include/llama.h": "8c1c645449a2e9946885ec39b82c4cded8ac3fc4a8f7261dd092c7848c038977",
    "src/llama-context.h": "6e14b32585c0de4bd18c7d41b59cc6fa6cd9c7acf35649312aaeae51d63747fc",
    "src/llama-context.cpp": "8b5c04b1096a8fc0dc8879bd58d299c51cb4e13e501494580b056e93c45a9d3f",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"Expected exactly one source anchor: {old[:100]!r}")
    return text.replace(old, new, 1)


def modify(name, text):
    if name == "include/llama.h":
        anchor = "    LLAMA_API uint32_t caps_llama_logits_width(const struct llama_context * ctx);"
        return once(text, anchor, anchor + '''

    // Private single-sequence, causal Qwen3VL ASR extension. output_limit must
    // be 1. Full vocabulary and the actual requested token computation are
    // unchanged; only worst-case dummy graph outputs are capped. Logical
    // decode batches requesting more output rows fail before KV/compute work.
    // CPU sampling remains supported; backend sampling/embeddings/encode and
    // adapters are not supported. Original initializers have no row limit.
    LLAMA_API struct llama_context * caps_llama_init_from_model_with_output_limit(
                     struct llama_model * model,
            struct llama_context_params   params,
                               uint32_t   output_limit);
    LLAMA_API uint32_t caps_llama_output_limit(const struct llama_context * ctx);
''')
    if name == "src/llama-context.h":
        text = once(text, "                  uint32_t logits_limit = 0);",
                    "                  uint32_t logits_limit = 0,\n                  uint32_t output_limit = 0);")
        text = once(text, "    uint32_t logits_width() const;",
                    "    uint32_t logits_width() const;\n    uint32_t output_limit() const;")
        return once(text, "    uint32_t n_outputs = 0; // number of actually-used outputs in the current ubatch or last logical batch",
                    "    uint32_t max_output_rows = 0; // private ASR contract; zero preserves upstream behavior\n    uint32_t n_outputs = 0; // number of actually-used outputs in the current ubatch or last logical batch")
    text = once(text, "              uint32_t logits_limit) :",
                "              uint32_t logits_limit,\n              uint32_t output_limit) :")
    text = once(text, "    cparams.logits_width = logits_limit ? logits_limit : model.vocab.n_tokens();",
                "    max_output_rows = output_limit;\n    cparams.logits_width = logits_limit ? logits_limit : model.vocab.n_tokens();")
    text = once(text, "uint32_t llama_context::logits_width() const {",
                "uint32_t llama_context::output_limit() const {\n    return max_output_rows;\n}\n\nuint32_t llama_context::logits_width() const {")
    # Row caps are a distinct contract from the existing vocabulary-width cap.
    for anchor, rejection in [
        ("void llama_context::set_embeddings(bool value) {", "value"),
        ("void llama_context::set_causal_attn(bool value) {", "!value"),
        ("bool llama_context::set_sampler(llama_seq_id seq_id, llama_sampler * sampler) {", "sampler"),
        ("void llama_context::set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {", "n_adapters"),
    ]:
        ret = "false" if anchor.startswith("bool ") else ""
        text = once(text, anchor, anchor + f'''
    if (max_output_rows && {rejection}) {{
        LLAMA_LOG_ERROR("%s: operation is not supported for single-output ASR contexts\\n", __func__);
        return {ret};
    }}'''.replace("return ;", "return;"))
    text = once(text, "                int32_t   il_end) {\n    LLAMA_LOG_DEBUG", '''                int32_t   il_end) {
    if (max_output_rows) {
        LLAMA_LOG_ERROR("%s: control vectors are not supported for single-output ASR contexts\\n", __func__);
        return false;
    }
    LLAMA_LOG_DEBUG''')
    text = once(text, "int llama_context::encode(const llama_batch & batch_inp) {", '''int llama_context::encode(const llama_batch & batch_inp) {
    if (max_output_rows) {
        LLAMA_LOG_ERROR("%s: encode is not supported for single-output ASR contexts\\n", __func__);
        return -1;
    }''')
    text = once(text, "    const uint32_t n_outputs_all = balloc->get_n_outputs();", '''    const uint32_t n_outputs_all = balloc->get_n_outputs();

    // The allocr only normalizes and validates the host batch. This guard is
    // before queued counters, scheduler reserve, memory_update, KV slot init,
    // graph execution and output buffer changes. Count the entire logical
    // batch, not each microbatch, to prevent a multi-ubatch bypass.
    if (max_output_rows && n_outputs_all > max_output_rows) {
        LLAMA_LOG_ERROR("%s: single-output ASR context rejects %u logical output rows (limit %u)\\n",
                __func__, n_outputs_all, max_output_rows);
        return -1;
    }''')
    # Also refuse multi-sequence use even with kv_unified, which otherwise
    # increases the batch allocator's allowed sequence count to LLAMA_MAX_SEQ.
    text = once(text, "    if (!balloc->init(batch_inp, vocab, memory.get(), n_embd, n_seq_max, output_all)) {", '''    if (max_output_rows) {
        uint32_t raw_outputs = 0;
        for (int32_t i = 0; i < batch_inp.n_tokens; ++i) {
            raw_outputs += batch_inp.logits && batch_inp.logits[i] != 0;
            if (raw_outputs > max_output_rows ||
                    (batch_inp.n_seq_id && batch_inp.n_seq_id[i] != 1) ||
                    (batch_inp.seq_id && (!batch_inp.seq_id[i] || batch_inp.seq_id[i][0] != 0))) {
                LLAMA_LOG_ERROR("%s: single-output ASR requires at most one output and sequence zero only\\n", __func__);
                return -1;
            }
        }
    }

    if (!balloc->init(batch_inp, vocab, memory.get(), n_embd, n_seq_max, output_all)) {''')
    text = once(text, "    ggml_backend_sched_reset(sched.get());\n\n    // when the scheduler is reset", '''    // Apply after rounding n_tokens: every dummy reservation path, including
    // FA probing, fallback and later scheduler rebuilds, obeys the same cap.
    // This never changes n_tokens/n_seqs or the real decode's output flags.
    if (max_output_rows) {
        n_outputs = std::min(n_outputs, max_output_rows);
        LLAMA_LOG_DEBUG("%s: private ASR reservation output rows capped to %u\\n", __func__, n_outputs);
    }

    ggml_backend_sched_reset(sched.get());

    // when the scheduler is reset''')
    text = once(text, "void llama_context::opt_init(struct llama_model * model, struct llama_opt_params lopt_params) {", '''void llama_context::opt_init(struct llama_model * model, struct llama_opt_params lopt_params) {
    GGML_ASSERT(max_output_rows == 0 && "single-output ASR contexts are inference-only");''')
    text = once(text, "                    uint32_t   logits_limit) {",
                "                    uint32_t   logits_limit,\n                    uint32_t   output_limit = 0) {")
    text = once(text, "new llama_context(*model, params, logits_limit)",
                "new llama_context(*model, params, logits_limit, output_limit)")
    anchor = "uint32_t caps_llama_logits_width(const llama_context * ctx) {"
    text = once(text, anchor, '''llama_context * caps_llama_init_from_model_with_output_limit(
        llama_model * model, llama_context_params params, uint32_t output_limit) {
    if (!model || model->arch != LLM_ARCH_QWEN3VL || !model->hparams.causal_attn ||
            output_limit != 1 || params.n_seq_max > 1 || params.embeddings ||
            params.n_samplers != 0 || params.samplers != nullptr ||
            (params.pooling_type != LLAMA_POOLING_TYPE_NONE && params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED) ||
            params.attention_type == LLAMA_ATTENTION_TYPE_NON_CAUSAL || !model->loras.empty()) {
        LLAMA_LOG_ERROR("%s: requires causal single-sequence Qwen3VL ASR, limit 1, no embeddings/adapters/backend samplers\\n", __func__);
        return nullptr;
    }
    LLAMA_LOG_INFO("%s: reserve at most %u output row; full vocabulary %u; actual tokens/KV unchanged\\n",
            __func__, output_limit, (uint32_t) model->vocab.n_tokens());
    return caps_init_context_impl(model, params, 0, output_limit);
}

uint32_t caps_llama_output_limit(const llama_context * ctx) {
    return ctx ? ctx->output_limit() : 0;
}

''' + anchor)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT,
                        help="Artifact directory; its native-src child is the isolated source copy")
    args = parser.parse_args()
    base = args.base_source.resolve(strict=True)
    out = args.output_dir.resolve()
    dest = out / "native-src"
    if dest.is_relative_to(base) or base.is_relative_to(dest):
        raise RuntimeError("Invalid independent-source boundary")
    prepared = {}
    checked_base = {}
    patch = []
    for name in FILES:
        original = (base / name).read_text("utf-8")
        actual = sha(original.encode("utf-8"))
        if actual != BASE_TEXT_SHA256[name]:
            raise RuntimeError(f"Unsupported base source {name}: normalized UTF-8/LF SHA256 "
                               f"{actual}, expected {BASE_TEXT_SHA256[name]}")
        checked_base[name] = actual
        updated = modify(name, original)
        prepared[name] = updated
        patch.extend(difflib.unified_diff(original.splitlines(True), updated.splitlines(True),
                                        fromfile="a/" + name, tofile="b/" + name))
    patch_text = "".join(patch)
    exported_patch = ROOT / "native/asr-single-output.patch"
    # Validate before copying source, stamping a receipt or writing any patch.
    # A checked-in/user-modified patch must never be silently replaced.
    if exported_patch.exists() and exported_patch.read_text("utf-8") != patch_text:
        raise RuntimeError("Refusing to overwrite an existing different exported patch: " + str(exported_patch))
    pattern = r"    struct llama_context_params \{.*?\n    \};"
    assert re.search(pattern, prepared["include/llama.h"], re.S)[0] == re.search(
        pattern, (base / "include/llama.h").read_text("utf-8"), re.S)[0]
    if not dest.exists():
        shutil.copytree(base, dest)
        for name, value in prepared.items():
            (dest / name).write_text(value, "utf-8", newline="\n")
    for name, value in prepared.items():
        if (dest / name).read_text("utf-8") != value:
            raise RuntimeError("Refusing to overwrite unexpected experimental changes: " + name)
    (out / "asr-single-output.patch").write_text(patch_text, "utf-8", newline="\n")
    if not exported_patch.exists():
        with exported_patch.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(patch_text)
    rows = []
    for path in sorted(base.rglob("*")):
        if path.is_file():
            relative = path.relative_to(base)
            rows.append({"path": relative.as_posix(), "base_sha256": sha(path.read_bytes()),
                         "candidate_sha256": sha((dest / relative).read_bytes())})
    receipt = {"upstream": "b9090/5757c4dcb178a01c85234a6db7503b19c9598873",
               "base_validation": {"scope": "Only the three patch-target files are authenticated; other files are recorded below",
                                   "encoding": "UTF-8 with normalized LF", "target_sha256": checked_base},
               "base_source": str(base), "candidate_source": str(dest), "files": rows,
               "changed": FILES, "public_context_params_unchanged": True, "models_loaded": 0}
    (out / "source-receipt.json").write_text(json.dumps(receipt, indent=2), "utf-8")
    print(json.dumps({"source": str(dest), "changed": FILES, "base_files": len(rows)}))


if __name__ == "__main__":
    main()
