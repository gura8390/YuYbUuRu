# CUDA 按需加载 cuBLAS

`cuda-delayload.patch` 适用于官方 llama.cpp **b9090 / `5757c4dcb178a01c85234a6db7503b19c9598873`**。它仅给 Windows MSVC 的 `ggml-cuda` 链接增加 `/DELAYLOAD:cublas64_12.dll` 和 `delayimp.lib`，不修改 CUDA 内核、模型、公开 ABI 或导出函数。补丁沿用 llama.cpp 的 MIT 许可。

原插件直接引用 cuBLAS 的 11 个函数；`cublasLt64_12.dll` 是 cuBLAS 的传递依赖。插件没有直接引用 Lt，因此不给插件添加无效的 Lt 延迟加载选项。两个 vendor DLL 都应保留在运行目录，按需加载不能代替 fallback 的实际验证。

## 可复现构建

在已有 CUDA 12.8.1、MSVC x64、Windows SDK、CMake、Ninja 的 Developer PowerShell 中，使用独立源码/构建/输出目录。以下相对目录是构建示例，不包含个人安装路径。

```powershell
$taskSource = Join-Path $PWD '.build/llama-b9090-cuda'
$taskBuild = Join-Path $PWD '.build/llama-b9090-cuda-build'
$taskPatch = Join-Path $PWD 'native/cuda-delayload.patch'
git clone --branch b9090 --depth 1 https://github.com/ggml-org/llama.cpp.git $taskSource
if ($LASTEXITCODE -ne 0) { throw 'Clone failed' }
if ((git -C $taskSource rev-parse HEAD).Trim() -ne '5757c4dcb178a01c85234a6db7503b19c9598873') { throw 'Unexpected source revision' }
git -C $taskSource apply --check $taskPatch
if ($LASTEXITCODE -ne 0) { throw 'Patch check failed' }
git -C $taskSource apply $taskPatch
if ($LASTEXITCODE -ne 0) { throw 'Patch failed' }

cmake -S $taskSource -B $taskBuild -G Ninja `
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON `
  -DGGML_NATIVE=OFF -DGGML_CPU=OFF -DGGML_CUDA=ON -DGGML_VULKAN=OFF `
  -DCMAKE_CUDA_ARCHITECTURES=120-real -DCMAKE_CUDA_RUNTIME_LIBRARY=Static `
  -DGGML_CUDA_GRAPHS=ON -DGGML_CUDA_NCCL=OFF -DGGML_CUDA_CUB_3DOT2=ON `
  -DGGML_CUDA_FORCE_MMQ=OFF -DGGML_CUDA_FORCE_CUBLAS=OFF `
  -DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF `
  -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF
if ($LASTEXITCODE -ne 0) { throw 'Configure failed' }
cmake --build $taskBuild --target ggml-cuda --parallel 2
if ($LASTEXITCODE -ne 0) { throw 'CUDA build failed' }
```

b9090 将上述 `120-real` 规范化为 `sm_120a`。这组选项对应当前原生对象的功能设置；CCCL 使用 3.2.0，需要时可给 CMake 增加 `FETCHCONTENT_SOURCE_DIR_CCCL` 指向已准备的官方 3.2.0 源目录。编译器、链接器、库版本和时间戳会影响 DLL 字节，不能用新构建的 hash 代替实际推理验收。

该构建仅针对本轮 SM120 的 NVIDIA RTX 5060 Laptop，不是可用于所有 NVIDIA 显卡的通用二进制。其他计算能力需按 CUDA/llama.cpp 支持的架构重新构建，并重新完成依赖、输出和稳定性验收；不要把架构专用 cubin 的兼容性外推到其他 GPU。

已有完成的同一构建时，最小实验应保留全部对象和原链接库，只重新链接到**新的输出目录**：在原链接响应文件末尾增加正确 MSVC x64 `delayimp.lib` 的路径，在 `link.exe` 参数中增加 `/DELAYLOAD:cublas64_12.dll`。保留 `/DLL /MACHINE:X64 /INCREMENTAL:NO /MANIFEST:EMBED,ID=2` 等原链接设置。中文路径的响应文件使用带 BOM 的 UTF-16；直接调用 `link.exe`，避免 CMake 包装器先按文本展开 UTF-16 响应文件。`rc.exe` 所在 SDK bin 需要在该构建子进程的 PATH 内。

这与候选实验的“原 136 个对象只重新链接”方法一致。验收应包括：对象 hash 前后一致；原 DLL 未覆盖；公开导出集相同；原 11 个 cuBLAS 名称完整转移至 delay import table；其他原依赖仍满足；实际 39 条 workload 的模块枚举确认 cuBLAS/Lt 是否始终未加载。

## 延迟加载的目录和失败边界

当前 worker 仅在自己的子进程 PATH 前加 native bin，保障后续 delayimp 调用的 DLL 搜索；不改变全局 PATH。初次 `LoadLibraryEx(..., LOAD_WITH_ALTERED_SEARCH_PATH)` 的一次性依赖搜索目录不会永久传递给日后的 delay helper。不能因为普通 MMQ workload 没调用 cuBLAS，就宣布 delayed cuBLAS fallback 已通过。

未来可采用一个**私有、绝对路径 delay hook**，本次仅设计，未实现或构建：

1. 新增一个小 C++ 文件，与插件一起链接 `delayimp`；不改 CUDA 内核或公开导出。
2. 仅处理 `dliNotePreLoadLibrary` 且 DLL basename 严格等于 `cublas64_12.dll` 的通知，其他 DLL 返回空值让标准 helper 处理。
3. 用插件内地址调用 `GetModuleHandleExW(FROM_ADDRESS | UNCHANGED_REFCOUNT)`，随后循环扩容调用 `GetModuleFileNameW` 得到插件的真实完整路径。不要从当前目录或用户输入推导路径。
4. 从插件所在目录构造 `cublas64_12.dll` 的绝对 Unicode 路径，以 `LoadLibraryExW` 和 `LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS` 加载。其 Lt 依赖由同目录搜索解决。
5. 返回该模块句柄给 delay helper；失败保留标准错误结果，记录 Windows 错误码，不私自改变工作目录或进程全局 DLL 搜索设置。路径必须覆盖中文、长路径和缺失依赖。

验证要真正触发原 cuBLAS 符号：独立 FP32 GEMM 或一个强制 cuBLAS 的控制样本，并检查成功前后 module list、计算结果及缺库失败行为。普通不使用 cuBLAS 的 39 项结果只验证按需未装载路径。

## 固定 token replay

`examples/decoder_replay_probe.rs` 直接使用当前 Rust NativeModel/NativeContext 包装，独立于正常 `Engine.decode`。接口：

```text
decoder_replay_probe --app-dir <isolated runtime> --backend cuda|vulkan
  --prefill-f32 <complete-prompt.f32-or.npy> --tokens-json <external-ids.json>
  --repeats 5 --n-ctx 2048|4096 --output <new-report.json>
```

构建/运行由测试窗口单独调度。每个后端使用不同 output，程序拒绝覆盖旧报告/日志。附属 `.native.stderr.txt` 保存 native loader、实际 GPU 名称、分层 offload 和各加载阶段；选定设备必须从日志得到证明，否则技术验收失败。

输入 `--prefill-f32` 是**完整 prompt**，不是 encoder 裸输出。支持无头 LE float32 `[N,1024]`，或 NPY v1/v2/v3 的 C-order `<f4`、形状 `[N,1024]` / `[1,N,1024]`；按 NPY 实际 header-length 字段确定 payload，不能固定剥除 80/128 字节。程序校验长度、有限值和 GGUF `<|im_start|>` / `<asr_text>` 首末 token 行，避免把形状相同的 encoder 输出误当完整 prompt。

原 root 的 `examples/engine_probe.rs` 导出的 `*.embedding.f32` 只包含 `transcribe_with_embedding` 返回的 `feature.data`；`clear_memory_probe` 只测 KV 清零，没有导出 prefill。可复用前者的音频特征，但需要按当前 `src/engine.rs::transcribe_with_embedding` 原样拼接一次：

```text
pre  = [im_start] + tokenize("system\n" + exact_context_or_default)
     + [im_end, im_start] + tokenize("user\n") + [audio_start]
post = [audio_end, im_end, im_start]
     + tokenize("assistant\n" + optional_normalized_language_prefix) + [asr_text]
full = GGUF token_embd rows(pre) + saved_encoder_feature + GGUF token_embd rows(post)
```

空 context 的默认值为 `You are a helpful assistant.`。语言自动识别不追加语言前缀；指定语言时使用与正常流程相同的规范化名字，前缀为 `language ` 加该名字。预先构造一次、按 LE f32 保存 full，再让两个进程读取同一文件。不要由两个后端各自生成一份“期望输入”。prefix/suffix 可用当前 `NativeModel.tokenize`、`gguf::EmbeddingTable::rows` 从同一只读 GGUF 构造，不需要运行编码器或修改正常 decode。

`--tokens-json` 必须是事先冻结的非空 i32 数组，来自同一次已保存的真实 native 解码记录 `raw.token_ids` / `metrics.token_ids`；不是文本分词后的近似结果。程序不采样、不自行生成 expected，也不因 EOS 提前退出，完整重放外给 IDs。总长度必须落在所选 context 内。

每轮清零一次；prefill 使用原 MRoPE 位置布局，token decode 使用生产 `decode_token` 的自动 KV 续接。请求 n_batch=4096、n_ubatch=512、两类线程数均为1。prefill 必须等 `get_logits_ith` 同步后停表；每个 token 同样同步再计时。generation_seconds 是这些逐 token 完成耗时之和；首 token、每512步末尾及最终 token 的全词表有限性/SHA 扫描放在计时区外，另留包含扫描的 loop wall。

这些边界与生产 `t_prefill/t_generate` 不同：生产 prefill 返回后的异步尾部可能计入 sampler/生成阶段，本探针显式计入 prefill 完成时间；不计 encoder、采样或全词表扫描。报告标记 `quality_equivalence=not evaluated`；输入 SHA 和相同 token 数只证明工作量相同，不能证明识别率或 CUDA/Vulkan logits 完全相同。
