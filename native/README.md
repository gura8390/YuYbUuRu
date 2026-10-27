# llama.cpp 原生补丁

CUDA 分支在以下 0.4.3 补丁之上增加 [ASR 单输出预留](asr-single-output.md) 和 [cuBLAS 延迟加载](cuda-delayload.md)。ASR 默认选择 CUDA；Vulkan 的构建步骤继续保留用于独立对照。原始普通初始化接口兼容性不代表应用默认配置兼容旧 DLL；本分支默认单输出和 4000 列对齐都要求对应扩展。

`aligner-logits.patch` 针对官方 [llama.cpp](https://github.com/ggml-org/llama.cpp) 的 **b9090**，提交 **`5757c4dcb178a01c85234a6db7503b19c9598873`**。它为只读、CPU Qwen3VL 文件对齐增加两个导出函数：`caps_llama_init_from_model_with_logits_limit` 和 `caps_llama_logits_width`。普通初始化入口继续输出完整词表；GPU 路线由应用配置显式选择。

`ggml-vulkan-cache.patch` 使用同一提交，为 Windows Vulkan 后端添加持久管线缓存及缓存失效后的无缓存重试，并将构建期 shader 编译器内部并发上限设为 2。补丁本身已经包含新增 `ggml-vulkan-cache.hpp`；仓库内的同名头文件用于独立审阅，不必再次手动复制。

补丁将文件对齐使用的原始 logits 保留为前 4000 列；模型词表、分词、嵌入及权重文件保持完整。受限上下文不支持采样、后端采样器、embeddings 模式、encode 或适配器。修改新增字段的位置是内部 `llama_cparams`，没有改变公开 `llama_context_params`；本项目对应的 Windows x64 公开结构仍为 136 字节。

## 从源码构建

安装官方 Visual Studio Build Tools 的 C++ x64 工具、Windows SDK、CMake 和 Ninja。在 **x64 Developer PowerShell** 中进入本仓库根目录，确认 `cl`、`cmake`、`ninja` 和 `git` 可用。以下使用正常安装的工具环境，不依赖个人 SDK 路径。

```powershell
$taskLlamaSource = Join-Path (Get-Location) '.build/llama-b9090'
$taskLlamaBuild = Join-Path (Get-Location) '.build/llama-b9090-build'
$taskLogitsPatch = Join-Path (Get-Location) 'native/aligner-logits.patch'
$taskVulkanPatch = Join-Path (Get-Location) 'native/ggml-vulkan-cache.patch'

git clone --branch b9090 --depth 1 https://github.com/ggml-org/llama.cpp.git $taskLlamaSource
if ($LASTEXITCODE -ne 0) { throw 'llama.cpp clone failed' }
$taskLlamaRevision = git -C $taskLlamaSource rev-parse HEAD
if ($taskLlamaRevision.Trim() -ne '5757c4dcb178a01c85234a6db7503b19c9598873') {
    throw 'The llama.cpp revision does not match the required base'
}
git -C $taskLlamaSource apply --check $taskLogitsPatch
if ($LASTEXITCODE -ne 0) { throw 'The logits patch does not apply cleanly' }
git -C $taskLlamaSource apply $taskLogitsPatch
if ($LASTEXITCODE -ne 0) { throw 'Applying the logits patch failed' }
git -C $taskLlamaSource apply --check $taskVulkanPatch
if ($LASTEXITCODE -ne 0) { throw 'The Vulkan cache patch does not apply cleanly' }
git -C $taskLlamaSource apply $taskVulkanPatch
if ($LASTEXITCODE -ne 0) { throw 'Applying the Vulkan cache patch failed' }

cmake -S $taskLlamaSource -B $taskLlamaBuild -G Ninja `
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON `
  -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF `
  -DGGML_CPU=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF `
  -DGGML_OPENCL=OFF -DGGML_SYCL=OFF `
  -DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_TESTS=OFF `
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF
if ($LASTEXITCODE -ne 0) { throw 'CMake configuration failed' }
cmake --build $taskLlamaBuild --target llama --parallel 2
if ($LASTEXITCODE -ne 0) { throw 'The llama target build failed' }

Get-FileHash (Join-Path $taskLlamaBuild 'bin/llama.dll') -Algorithm SHA256
```

共享库和动态后端装载开启；此次构建关闭 CPU、CUDA、Vulkan、OpenCL、SYCL 后端目标，因为运行目录继续使用同一 b9090 构建系列的原有后端。构建产物的机器码与 SHA256 会受编译器和工具版本影响。

上述步骤生成对齐补丁所需的 `bin/llama.dll`。Vulkan 缓存还需构建下面的后端目标。

## 构建 Vulkan 后端

从 [LunarG 官方站点](https://vulkan.lunarg.com/sdk/home#windows) 安装完整 Windows Vulkan SDK，安装后重新打开 x64 Developer PowerShell，使 `VULKAN_SDK`、Vulkan 头文件、库及 `glslc` 可用。继续使用已经应用两个补丁的同一源码目录，单独配置 Vulkan 构建目录：

```powershell
$taskVulkanBuild = Join-Path (Get-Location) '.build/llama-b9090-vulkan-build'
if (-not $env:VULKAN_SDK) { throw 'Install the Windows Vulkan SDK and reopen Developer PowerShell' }

cmake -S $taskLlamaSource -B $taskVulkanBuild -G Ninja `
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON `
  -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF `
  -DGGML_CPU=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=ON `
  -DGGML_OPENCL=OFF -DGGML_SYCL=OFF `
  -DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_TESTS=OFF `
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF
if ($LASTEXITCODE -ne 0) { throw 'Vulkan CMake configuration failed' }
cmake --build $taskVulkanBuild --target ggml-vulkan --parallel 2
if ($LASTEXITCODE -ne 0) { throw 'The ggml-vulkan target build failed' }

Get-FileHash (Join-Path $taskVulkanBuild 'bin/ggml-vulkan.dll') -Algorithm SHA256
```

`--parallel 2` 限制 CMake 外层构建进程并发；补丁将构建期 `vulkan-shaders-gen` 内部调用 `glslc` 的并发上限设为 2，属于不同的控制项。运行时 Vk 管线创建线程池仍使用原有 `hardware_concurrency`，此补丁没有改变它。SDK、`glslc` 与编译器版本会影响生成的 shader 和 DLL 字节，不能用源码一致直接代替实际推理验证。

停止本应用及其 worker、保存原文件后，仅将生成的 `llama.dll` 和 `ggml-vulkan.dll` 放入运行目录的 `core/server/engines/llama/bin/`。保留匹配 b9090 的原 `ggml.dll`、`ggml-base.dll`、CPU 后端、OpenMP 及其它依赖 DLL；不要把此次构建的其余 ggml DLL 一并覆盖，也不要混用不同提交的后端。本源码仓库不附带这些二进制。

## 缓存行为

持久缓存仅实现于 Windows。进行 Vulkan 对照时设置 `gpu_backend=vulkan` 和 `vulkan_pipeline_cache=true`；本 CUDA 分支默认关闭 Vulkan 缓存。应用只在自己创建的 Vulkan worker 环境中设置 `GGML_VK_PIPELINE_CACHE_DIR`，目录为 `app_dir/cache/vulkan`，不会修改全局环境。设置为 `false` 时从该子进程环境移除此变量，恢复无持久缓存路径。

缓存按设备、驱动版本和 Vulkan pipeline cache UUID 区分，校验长度、格式和 CRC。首次空缓存、驱动变化或缓存不可用时仍需编译管线。缓存读写失败按未命中处理；缓存参与的管线创建失败时，对该设备禁用缓存并按原无缓存路径重试。缓存文件属于本机派生产物，不随源码发布。

只有显式设置 `GGML_VK_PIPELINE_STATS` 才启用 Vulkan 管线执行统计扩展，检查的是变量是否存在，因此设置为 `0` 也会开启。默认不设置。单独关闭统计在本机对照中没有测得性能收益，不能将其作为 prefill 提速成果。

`settings.example.json` 的 `aligner_logits_limit=4000` 需要上述扩展。旧 DLL 缺少扩展时会明确失败。使用普通完整词表 DLL 时应设置 `aligner_logits_limit=0` 和 `asr_single_output=false`；`aligner_dynamic_context=false` 与 `file_context_handoff=false` 可恢复原固定上下文及保留策略。本分支应用 Settings 缺字段时采用 `true/4000/true` 的对齐优化默认值；底层 EngineOptions 仍保留旧诊断接口默认值。

## 许可

llama.cpp 保留 **The ggml authors** 的 MIT 版权，原许可全文见 [llama.cpp-LICENSE](llama.cpp-LICENSE)。两个补丁沿用 MIT 许可；发布或再分发构建结果时须携带原作者版权及许可。原模型、后端、驱动和其它运行依赖继续适用各自许可。
