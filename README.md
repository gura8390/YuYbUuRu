# 语音输入 · YuYbUuRu

Windows x64 离线语音输入工具，当前版本 **0.4.3**。应用层使用 Rust，日常运行无需 Python；推理使用 Qwen3-ASR-0.6B、ONNX Runtime 和 llama.cpp 原生库。

保留三个功能：实时听写、热词替换、音视频文件转录。长按 CapsLock 或鼠标 X2 超过 0.3 秒开始听写，松开后输入；短按补发原键。托盘提供热词编辑、文件转录及退出。文件输出 TXT、JSON、SRT，默认保留已有同名输出。

本项目参考并移植了 [CapsWriter-Offline](https://github.com/HaujetZhao/CapsWriter-Offline) 的应用行为、热词、格式化、合并和字幕算法，保留原作者 **Haujet Zhao** 的 MIT 版权及第三方声明。这是独立的 Rust 重构仓库，不是上游的 GitHub Fork。

## 构建

需要 Windows x64、Rust stable MSVC、C++ Build Tools 和 Windows SDK。

```powershell
# 已安装完整 Windows SDK 时，直接构建
.\build.ps1
.\build.ps1 -Tests

# 缺少 SDK 导入库时可先执行；仅下载到本项目 .build，不改全局环境
python .\tools\bootstrap_sdk.py
```

产物为 `target\release\yuybuuru.exe`。Python 3.11+ 仅用于可选的 SDK 准备或字典再生成，不参与语音推理。

## 运行目录

仓库只发布源码、通用词典、许可证和合成测试数据，模型、原生 DLL 和 FFmpeg 需另外准备。可复用合法取得的兼容 CapsWriter 运行库，保留它们的许可与依赖文件；不要随意混用不同 llama ABI 的 DLL。

```text
runtime/
  start_client.exe                 # 将构建产物复制为此名称
  settings.json                   # 从 settings.example.json 复制
  hot.txt                         # 个人热词；仓库示例不含个人资料
  hot-rule.txt                    # 可选的替换规则
  ffmpeg.exe                      # 连同其所需 DLL
  core/server/engines/llama/bin/   # 兼容的 llama、ggml、CPU/Vulkan、libomp DLL
  internal/onnxruntime/capi/       # onnxruntime.dll 与配套依赖
  internal/_sounddevice_data/portaudio-binaries/libportaudio64bit.dll
  models/Qwen3-ASR/Qwen3-ASR-0.6B/
  models/Qwen3-ForcedAligner/Qwen3-ForcedAligner-0.6B/
```

ASR 目录包含 `qwen3_asr_encoder_frontend.int4.onnx`、`qwen3_asr_encoder_backend.int4.onnx`、`qwen3_asr_llm.q4_k.gguf`。对齐目录使用对应的 `qwen3_aligner_encoder_frontend.int4.onnx`、`qwen3_aligner_encoder_backend.int4.onnx`、`qwen3_aligner_llm.q4_k.gguf`；模型目录中的其它外部权重文件也须完整保留。

```powershell
New-Item -ItemType Directory -Path '.\runtime' -Force
Copy-Item '.\target\release\yuybuuru.exe' '.\runtime\start_client.exe'
Copy-Item '.\settings.example.json' '.\runtime\settings.json'
Copy-Item '.\assets\text\hot_default.txt' '.\runtime\hot.txt'
Copy-Item '.\assets\text\rules_default.txt' '.\runtime\hot-rule.txt'
# 将上述原生运行库、FFmpeg 与模型准备完整后启动
.\runtime\start_client.exe
.\runtime\start_client.exe transcribe '.\sample.wav' --output-dir '.\output'
# 明确需要覆盖已有输出时增加 --overwrite
```

也可用 `--app-dir` 指定运行目录、`--config` 指定设置文件。发行配置的 ASR 编码器线程为 8/1，Vulkan 解码器线程为 1/1；文件对齐器沿用独立 CPU 配置。低层诊断接口保留原 16 线程默认值。

## 热词与资源管理

`hot.txt` 每行一个目标词，使用 `|` 添加发音别名，例如 `Codex|扣德克斯`。热词和规则在文件修改后重新加载。仓库内置的默认示例均为通用词语。

模型闲置 120 秒后退出整个推理进程，下次请求时重建；文件对齐器闲置 10 秒释放。采集音频队列有 1 MiB 分配预算，超额或明确丢帧时废弃整句，避免缺帧文本被当作成功输出。文件使用一个工作线程，最多排队四条路径。

0.4.1 将 ASR 和文件对齐器各自独立的编码器、语言模型及上下文初始化并行，加载完成后回到原有串行推理。模型、量化、线程设置和空闲卸载策略保持原样。设置 `YYUR_PROFILE_LOAD=1` 可在 worker 日志中记录各加载阶段的耗时；默认关闭。

0.4.2 保留 Vulkan 推理、ASR 8/1 线程和系统默认 CPU 调度。文件对齐器默认关闭 ONNX Runtime CPU arena（`aligner_cpu_mem_arena=false`），减少文件转录期间的保留内存；模型、量化和空闲卸载策略沿用原设置。

0.4.3 的示例配置启用文件对齐器动态上下文（`aligner_dynamic_context=true`）、4000 列时间戳 logits（`aligner_logits_limit=4000`）及文件上下文交接（`file_context_handoff=true`）。动态上下文按实际提示长度分配；交接在 ASR 与对齐阶段间释放解码上下文，继续保留模型权重和编码器，以降低同时保留的内存。4000 列模式需要按 [原生库构建说明](native/README.md) 应用补丁并重建 `llama.dll`。旧配置缺少这三个字段时仍使用 `false/0/false`；将交接关闭可保留上下文，减少重复构建的耗时。

Windows Vulkan 持久管线缓存由配套 `ggml-vulkan.dll` 补丁提供，`vulkan_pipeline_cache=true` 默认开启。缓存保存在运行目录的 `cache/vulkan`，只通过本应用自己创建的 worker 环境传递；关闭此设置可恢复无持久缓存路径。首次空缓存、驱动变化或缓存不可用时仍需编译管线。两项原生补丁及其独立构建目标见 [原生库构建说明](native/README.md)。

文件中间分段只返回轻量确认，完整累计结果保留在 worker。请求 JSON 上限 1 MiB，最终回复上限 64 MiB；超限明确报错。首次完整初始化 KV 缓存，后续成功调用仅重置状态；原生解码异常后，下次恢复完整清零。

## 测量与验证范围

一组同机、相同 29 条输入、每版三轮的历史服务/推理进程树对照：最初 1.7B 原版到 0.4.0，热态私有提交约 5814.1 → 1271.2 MiB（下降 78.14%），工作集约 1493.6 → 177.1 MiB（下降 88.14%），响应中位约 273.5 → 150.8 ms（下降 44.85%）。

这些结果同时包含模型、量化、线程、库构建、协议和生命周期变化，不能全部归因于 Rust，也不是峰值、全系统物理 RAM 或显存保证。0.4 相对 0.3 的响应中位改善约 5.87%，p95 增加约 1.27%，热态提交基本持平。

两版 Rust 在固定输入下输出一致；0.6B 与原版 1.7B 的输出存在差异，未据此认定识别率等同。物理按键、实际粘贴、设备拔插和所有第三方窗口组合未完成完整人工验收。

0.4.0 到 0.4.1 的同机冷进程对照：保留 Windows 文件缓存、固定 5.592 秒中文样本及种子，模型就绪中位约 0.991 → 0.780 秒（下降 21.28%，每版 6 轮）；首个听写请求含加载约 1.132 → 0.951 秒（下降 16.00%，每版 3 轮）；首个文件请求含加载与对齐约 2.166 → 1.649 秒（下降 23.87%，每版 3 轮）。两版共 48 条结果的五个字段精确一致。热态提交基本持平。新版预热组保留一次 2.380 秒首次完成异常，原因未确定；中位数改善不保证每次或最坏启动时间更低，也不代表重启后冷盘或长文件总耗时。

0.4.1 到 0.4.2 的三轮长短文件交替对照（每轮七个独立完整 PCM 请求，共 253.25 秒音频）：服务/worker 树的 25ms 采样峰值私有提交中位约 3646.1 → 3046.3 MiB（减少 599.9 MiB，16.45%），末短提交约 3440.4 → 2812.2 MiB（减少 18.26%）。七条总耗时约 19.958 → 20.596 秒（增加 3.20%），采用 RAM 优先的取舍；42 条结果五字段全部一致。另两版各 90 条听写结果一致，听写批次耗时中位增加约 1.20%，冷请求增加约 0.46%，没有宣称每个指标都改善。

上述 0.4.2 对照保留 Vulkan、ASR 8/1 线程和默认调度，只默认关闭文件对齐器 CPU arena；`aligner_cpu_mem_arena=true` 可恢复原内存池。精确版本的 CUDA 12.8 llama 解码器也已实际测试（ONNX 编码器仍用 CPU），其单轮提交更高且 11/39 固定结果与 Vulkan 不同，未采用；差异不等同于 11 条识别错误。CUDA 依赖不加入运行包。采样峰值是观测下界，非瞬时真峰或全系统内存；本轮百分比不可与历史总减幅相加。

0.4.3 最终原生方案的文件对照使用同机、每组三轮新进程，每轮七个独立完整 PCM 请求，保留 Vulkan、ASR 8/1 和已关闭的文件对齐器 CPU arena。采用方案同时启用动态上下文、4000 列 logits、上下文交接及配套 Vulkan 缓存。下表为三轮中位数，内存单位为 MiB，CPU 与耗时单位为秒：

| 文件配置 | 服务/worker 树采样峰值私有提交 | 末短私有提交 | 序列 CPU 时间 | 七请求总耗时 |
|---|---:|---:|---:|---:|
| 原固定上下文、完整词表 | 3047.8 | 2813.7 | 370.516 | 21.067 |
| 0.4.3 最终原生方案 | 2006.1 | 1406.0 | 341.469 | 20.995 |

最终方案的采样峰值减少 34.18%，末短提交减少 50.03%，CPU 时间减少 7.84%；总耗时减少 0.34%，视为基本持平。两组三轮共 42 条文件结果精确一致。该表使用最终原生库整体对照，未混入探索阶段的交接耗时结果。

Vulkan 缓存另有每模式三轮对照：保留系统及驱动缓存，每轮采用新的 EXE 身份，并为开启组预置本应用缓存。在这一驱动缓存未命中的测试条件下，首次原生 prefill 中位约 0.890 → 0.065 秒（减少 92.64%），首次完整请求约 1.810 → 0.968 秒（减少 46.51%），九请求总耗时约 9.758 → 7.317 秒（减少 25.01%）。空应用缓存的单轮首次 prefill 仍约 0.901 秒。另一个旧 EXE 身份、驱动缓存原已命中的单轮描述性对照约 0.013 → 0.046 秒，多约 33ms；它不是统计结论，持久缓存也不保证所有冷启动都更快。这些实验保留系统缓存，不是重启后冷盘测试。

最终原生库的 39 条 ASR 原始结果及五个输出字段精确一致；独立对齐检查保留了前 4000 列 logits 的逐 float 字节一致性。缓存管线创建失败、CRC 损坏、超过 64MiB 和不可写目录四类实际故障验证均通过。另有八请求混合模式验证，包含一次预期语言参数错误及其后恢复，六项指定 TXT/JSON/SRT 导出的 SHA256 一致。

进程树峰值来自目标 25ms 间隔采样，包含加载，仍是瞬时峰值的观测下界，排除桌面进程、GPU 专用内存及驱动。末短提交只对应本组最后一个短请求。文件性能组验证的是独立 PCM 请求；指定混合模式与导出检查也不覆盖所有流式分段边界、媒体文件或第三方窗口组合。

发布树使用通用合成热词、格式化、合并、字幕和数值参考，不包含私人录音、个人热词或本机测量资料。完整设备/模型烟测默认忽略，普通测试不启动模型或操作设备。

## 许可证

[MIT](LICENSE)。原始 CapsWriter 许可、拼音数据、参考算法及 Rust 依赖许可见 [THIRD-PARTY-NOTICES.txt](THIRD-PARTY-NOTICES.txt)、`assets/text/README.txt` 和 `assets/licenses/`。模型、FFmpeg、PortAudio、ONNX Runtime、llama.cpp 及各 DLL 保留各自许可证。
