# 语音输入 · YuYbUuRu

Windows x64 离线语音输入工具，当前版本 **0.4.0**。应用层使用 Rust，日常运行无需 Python；推理使用 Qwen3-ASR-0.6B、ONNX Runtime 和 llama.cpp 原生库。

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

文件中间分段只返回轻量确认，完整累计结果保留在 worker。请求 JSON 上限 1 MiB，最终回复上限 64 MiB；超限明确报错。首次完整初始化 KV 缓存，后续成功调用仅重置状态；原生解码异常后，下次恢复完整清零。

## 测量与验证范围

一组同机、相同 29 条输入、每版三轮的服务/推理进程树对照：最初 1.7B 原版到本版，热态私有提交约 5814.1 → 1271.2 MiB（下降 78.14%），工作集约 1493.6 → 177.1 MiB（下降 88.14%），响应中位约 273.5 → 150.8 ms（下降 44.85%）。

这些结果同时包含模型、量化、线程、库构建、协议和生命周期变化，不能全部归因于 Rust，也不是峰值、全系统物理 RAM 或显存保证。0.4 相对 0.3 的响应中位改善约 5.87%，p95 增加约 1.27%，热态提交基本持平。

两版 Rust 在固定输入下输出一致；0.6B 与原版 1.7B 的输出存在差异，未据此认定识别率等同。物理按键、实际粘贴、设备拔插和所有第三方窗口组合未完成完整人工验收。

发布树使用通用合成热词、格式化、合并、字幕和数值参考，不包含私人录音、个人热词或本机测量资料。完整设备/模型烟测默认忽略，普通测试不启动模型或操作设备。

## 许可证

[MIT](LICENSE)。原始 CapsWriter 许可、拼音数据、参考算法及 Rust 依赖许可见 [THIRD-PARTY-NOTICES.txt](THIRD-PARTY-NOTICES.txt)、`assets/text/README.txt` 和 `assets/licenses/`。模型、FFmpeg、PortAudio、ONNX Runtime、llama.cpp 及各 DLL 保留各自许可证。
