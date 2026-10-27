# ASR single output reservation experiment

This overlay applies to the accepted `aligner-logits.patch` source based on
llama.cpp b9090, commit `5757c4dcb178a01c85234a6db7503b19c9598873`.
It changes only `include/llama.h`, `src/llama-context.h` and
`src/llama-context.cpp`. Public `llama_context_params` stays byte-for-byte intact.
The GGML backend ABI and the Qwen3VL computation source are untouched.

Upstream `sched_reserve()` requests 512 output rows for its dummy prompt graph
when `n_ubatch=512`. The shipped ASR model reports 151936 vocabulary entries;
its 512-row FP32 head alone is 296.75 MiB. The aligner is a different model with
152064 entries. Code and probes always query the loaded model's vocabulary.
Production
ASR requests one last output row for prefill and one row per decoded token. The
private context caps only dummy reservations to one output, after any token
count rounding. It preserves token batching, attention, positions, KV types,
vocabulary, tensor weights, and all real output flags. Actual memory and speed
need measurement: the 296.17 MiB reduction of this ASR dummy head is a theoretical
tensor-size difference, not a measured reduction of the complete buffer.

Private exports:

```c
llama_context * caps_llama_init_from_model_with_output_limit(
    llama_model *, llama_context_params, uint32_t output_limit);
uint32_t caps_llama_output_limit(const llama_context *);
```

The initializer accepts only `output_limit=1`, causal Qwen3VL, one sequence,
no embeddings, backend samplers or adapters. It always uses the complete
vocabulary. A logical batch with two or more requested rows is rejected before
KV mutation or graph computation, including rows in different microbatches.
Explicit multi-sequence batches are also rejected, even with unified KV.
Encode, training and mode switches outside this contract are refused. Ordinary
CPU sampling still reads the complete vocabulary.

The legacy initializer and the CPU 4000-vocabulary alignment initializer retain
an output limit of zero. No row cap is applied to them. Rust enables the new
initializer only with `asr_single_output: true`. Application `Settings` now
defaults this field to true, including old settings files which omit it. The
low-level `EngineOptions::default()` remains false for library compatibility.
Both eager and recreated ASR contexts use the selected policy. CPU alignment
is unchanged. Application upgrades must ship the matching DLL with both new
private exports; using an older DLL with the setting enabled fails explicitly.
Set `asr_single_output: false` to use the ordinary initializer with an older DLL.

Reproduce in an isolated working copy. First extract the pinned upstream
source and apply `native/aligner-logits.patch` to that copy as described in
`native/README.md`. Pass this prepared copy as `--base-source`; no sibling
project or private evidence directory is required.

Before creating the isolated source or writing a receipt, the preparation script
authenticates the three patch-target files against pinned SHA256 values for
b9090 plus the accepted aligner overlay. Hashes use UTF-8 text with normalized
LF, so ordinary CRLF checkout conversion is accepted. Other source files are
recorded in the receipt, rather than authenticated by this three-file gate.
An existing different `native/asr-single-output.patch` causes a failure before
any source copy or version stamp; an identical exported patch is left untouched.
The generated patch is also saved inside the selected artifact directory.

```powershell
python tools/prepare_asr_single_output.py --base-source <accepted-limited-native-src> --output-dir <artifact-dir>
powershell -NoProfile -ExecutionPolicy Bypass -File tools/build_asr_single_output.ps1 -NativeSource <artifact-dir>/native-src -BuildDir <build-dir> -SdkInclude <sdk-include-version-dir> -SdkLib <sdk-lib-version-dir> -SdkBin <sdk-bin-version-x64-dir> -VcVars <vcvars64.bat> -CMake <cmake.exe> -Ninja <ninja.exe> -Jobs 2
python tools/verify_asr_single_output.py --candidate <build-dir>/bin/llama.dll --original-bin <original-runtime-native-bin> --output-dir <artifact-dir> --source-dir <artifact-dir>/native-src --dumpbin <dumpbin.exe>
```

The build script accepts MSVC, CMake, Ninja and Windows SDK locations explicitly.
`-SdkInclude` contains `um/Windows.h`, `ucrt/stdlib.h` and the other include
subdirectories; `-SdkLib` contains `um/x64` and `ucrt/x64`; `-SdkBin` contains
`rc.exe`. An installed SDK and an extracted portable SDK both work. `-VcVars`
may be omitted in an MSVC developer shell; CMake/Ninja/dumpbin may be omitted
when available on that shell's PATH. SDK paths are always explicit. Source and
build defaults belong to this project (`evidence/cuda-route/single-output/native-src`
and `.build/asr-single-output`); the verifier defaults to that build's DLL.
For toolchains affected by non-ASCII response-file paths, supply ASCII source,
SDK and build paths or caller-created junction aliases. No global aliases are
created by the scripts.
Keep artifact/source paths short on Windows: the full upstream tree contains
deep web UI paths which can exceed the system's enabled path-length limit.

It builds only the `llama` target with runtime backend loading enabled and
all compute backends disabled. Its newly built GGML dependency DLLs are build
inputs only. Runtime verification retains the original GGML/CPU/CUDA plugins.
The ignored `evidence/cuda-route/single-output` directory contains per-file
source hashes, full logs, the candidate DLL, import/export receipts and
isolated no-model checks. These checks do not establish recognition quality
or GPU memory savings. Default-full control, all accepted quality fixtures,
file transcription/alignment and repeated memory/performance comparisons are
required before selecting this candidate.

Generic fixed-workload diagnostics:

```powershell
cargo build --release --example make_decoder_fixture --example decoder_replay_probe
target/release/examples/make_decoder_fixture.exe --app-dir <vulkan-runtime> --embedding-f32 <encoder-output.f32> --tokens-json <accepted-vulkan-token-array.json> --output-dir <new-fixture-dir>
target/release/examples/decoder_replay_probe.exe --app-dir <candidate-runtime> --backend cuda --prefill-f32 <fixture-dir>/prefill.f32 --tokens-json <fixture-dir>/tokens.json --n-ctx 4096 --repeats 5 --output <new-report.json>
python tools/cuda_blas_smoke.py --native-bin <delay-cublas-runtime-bin> --output <new-smoke-report.json>
```

The fixture builder uses CPU-only native model loading for original tokenization
and embedding dequantization, and creates no inference context. Caller-supplied
accepted tokens are preserved rather than generated. Replay synchronizes prefill
and each decode before stopping its timer, with full-vocabulary finite/hash scans
outside the primary per-token timing. Its timing boundary differs from production
`t_prefill/t_generate`, where pending work can complete in sampling. The cuBLAS
smoke runs only a small known F32 identity matrix multiplication, then proves
actual private cuBLAS/Lt loading and checks the numeric result. It does not test
every vendor operation merely by resolving all original 11 symbols.

The 151936 ASR vocabulary size above describes the shipped model only. Generic
fixture/replay/guard probes obtain the vocabulary size from the loaded model's
native metadata; they do not hard-code 151936 or the aligner's 152064 entries.

Upstream llama.cpp is MIT licensed; see `llama.cpp-LICENSE`.
