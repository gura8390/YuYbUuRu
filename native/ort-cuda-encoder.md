# ORT CUDA encoder experiment

This branch keeps `encoder_provider: "cpu"` as the default. The optional values
`cuda_backend` and `cuda_all` affect the ASR encoder only. Alignment explicitly
uses the CPU EP, including when a GPU-capable ORT core is selected.

## Isolated runtime and controls

The installed CPU ORT reports C API version **1.24.4**, file version
`1.24.20260316.2.2d92497`. The selectively extracted official Windows
`onnxruntime-gpu==1.24.4` DLLs have the same file version. Source tag
[`v1.24.4`](https://github.com/microsoft/onnxruntime/tree/v1.24.4) resolves to
`2d924974ef147392ced8409d36bd6d2e7fcc8a74`.

Use `tools/prepare_ort_cuda_runtime.py`,
`tools/prepare_ort_cuda_dependencies.py` and `tools/inspect_ort_cuda_closure.py`
for selective ZIP member retrieval, member CRC/SHA verification and static PE
closure. Full archive hashes are only published metadata when the full archive
was not downloaded. These scripts do not create ORT sessions or initialize CUDA.

`tools/stage_ort_cuda_encoder.py` only adds a new `runtime-ort-gpu/capi`
directory inside an existing isolated research app. It refuses any target
outside this checkout's `evidence/runtimes`. It does not change the CPU `internal`
junction, settings, or the installed app. Staged DLLs are immutable links/copies.

Run separate fresh workers in this order, keeping the same decoder, input,
seeds, thread policy and model files:

1. Original CPU ORT core + `encoder_provider: "cpu"`.
2. GPU ORT core + `encoder_provider: "cpu"` (core-build control).
3. GPU ORT core + `encoder_provider: "cuda_backend"`.
4. GPU ORT core + `encoder_provider: "cuda_all"`.

The optional `onnx_dll` setting selects the DLL; relative paths resolve against
`app_dir`, for example `runtime-ort-gpu/capi/onnxruntime.dll`. One process can
load only one ORT DLL. The owned worker receives a private PATH containing that
directory and the CUDA decoder directory. A diagnostic that starts `worker`
directly must set this child-process PATH itself.

## Deliberately partial dependency preparation

The exact ORT provider imports are cublas, cublasLt, cudart, cuFFT and the cuDNN
top-level DLL. No curand import was present in this artifact. Reuse the decoder's
matching cublas/cublasLt/cudart DLLs; do not download a complete toolkit.

Approved additional cuDNN 9.7.1.26 members are only the top-level loader, graph,
ops and cnn. Adv, engines and heuristic DLLs are intentionally not staged yet.
cuDNN loads sublibraries lazily, so a complete static PE import closure does not
prove that a CUDA EP session or convolution will work. An embedded sublibrary
name likewise does not prove this particular model will request it. Preserve
the actual first session error before requesting more dependencies.

In the completed experiment, backend-only CUDA inference succeeded and actual
heavy-node CUDA placement was verified. The full frontend's first Conv failed
because `cudnn_engines_precompiled64_9.dll` was absent. An explicit optional
`--frontend-engines` preparation path retrieves precompiled, runtime-compiled
and heuristic members only. That additional download was stopped for low
expected value, with its incomplete CRC-unverified partial preserved locally;
it was never staged. Full-frontend GPU speed, memory and quality remain
unmeasured. The backend-only candidate changed raw output on 10/39 fixtures and
increased private commitment, so CPU remains the default.

The selected expanded ORT + cuFFT + these cuDNN files occupy 727,863,056 logical
bytes; existing cublas/cublasLt/cudart add 788,957,696 linked logical bytes.
File sizes and mapped read-only DLL sections are not private RAM measurements.

The selected cuFFT DLL is a particular memory risk: PE inspection finds a
read/write `.data` section with virtual size `0x10535174` (about 261.2 MiB),
raw size `0x102c0400`. This alone is not a private-memory measurement. ORT's
provider statically imports only `cufftCreate`, `cufftDestroy`, `cufftSetStream`,
`cufftXtExec` and `cufftXtMakePlanMany`; these ASR graphs contain no FFT node.
Measure loaded modules and private memory before considering any separate
delay-load change. No cuFFT proxy or vendor DLL patch is implemented here.

NVIDIA's [cuDNN 9.7.1 support matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.7.1/reference/support-matrix.html)
lists Blackwell compute capability 12.0 with CUDA 12.8 and Windows driver at
least 570.65. Static compatibility is not an SM120 execution test.

## Memory, numerical and performance boundaries

The CUDA EP is explicitly registered with failure propagation, TF32 disabled,
`kSameAsRequested`, heuristic convolution search and max convolution workspace
disabled. `encoder_cuda_mem_limit_mb` defaults to 512 MiB per CUDA session
(FE and BE each have a separate arena in `cuda_all`). As described in the
[CUDA EP documentation](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html),
the memory limit applies to its arena, not total VRAM, driver allocations or
CPU private DLL data. Convolution workspace options are not a total device
memory cap. Existing CPU memory-pattern/prepacking/device-initializer options
remain independently configurable. Arena shrinkage is opt-in, on the backend
only; GPU shrinkage uses the documented `gpu:0` device notation.

Even backend-only CUDA creates the CUDA EP's per-thread cuBLAS, cuBLASLt and
cuDNN handles ([v1.24.4 source](https://github.com/microsoft/onnxruntime/blob/v1.24.4/onnxruntime/core/providers/cuda/cuda_execution_provider.cc)).
It can reintroduce cuBLASLt private memory avoided by the decoder's delay-load
optimization. Compare against the current delayed-CUBLAS CPU-encoder candidate.

The current models take FP32 inputs. The int4
[`MatMulNBits` CUDA implementation](https://github.com/microsoft/onnxruntime/blob/v1.24.4/onnxruntime/contrib_ops/cuda/quantization/matmul_nbits.cc)
can dequantize packed weights into floating-point scratch; int4 disk size is
not a guarantee of equally small VRAM. FE operates in 100-frame CPU-prepared
chunks and returns each chunk to a CPU vector. BE-only avoids adding GPU
transfers to each frontend chunk. The final BE embedding also returns to CPU
before the native decoder consumes it. No zero-copy claim is made here.

TF32 disabled is a requested precision policy, not numerical equivalence
evidence. The v1.24.4 cuDNN
[convolution implementation](https://github.com/microsoft/onnxruntime/blob/v1.24.4/onnxruntime/core/providers/cuda/nn/conv.cc)
can retry a failed frontend plan with TF32 permitted. Compare embeddings and
the fixed token/result acceptance fields on the full fixture set before
accepting any GPU encoder result.

## Actual node placement profiling

Set `YYUR_ORT_PROFILE_DIR` only for a diagnostic run. FE and BE get distinct
PID/model prefixes. `AudioEncoder::drop` explicitly finalizes both profiles.
A killed worker may leave empty or incomplete output; that is not accepted
placement evidence. Shut down naturally or explicitly drop the engine.

`tools/summarize_ort_profile.py` groups executed Node kernel-time events by
actual `args.provider` and operator. CPU shape nodes may be intentional, while
heavy operators must be inspected separately. Merely seeing CUDA registered
or one CUDA node does not prove all heavy computation is on GPU.

ORT's [profiling documentation](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html)
distinguishes ordinary host-side node events from CUDA device-kernel profiling,
which requires CUPTI and a profiling-enabled ORT build. Host durations and their
interval union are not CUDA device-kernel timing. Keep profiling runs separate
from the final unprofiled performance/memory comparison.
