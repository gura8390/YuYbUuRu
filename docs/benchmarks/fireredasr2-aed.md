# FireRedASR2-AED 本机对比测试

测试日期为 2026 年 11 月 6 日。对照对象为 YuYbUuRu v0.4.5 的 Qwen3-ASR 1.7B 配置。FireRed 官方 BF16 路线在模型加载后的短句测试中更快，52.60 秒音频的转写耗时和资源峰值更高。本次继续使用 Qwen3-ASR 1.7B。

## 环境与方法

同一台 Windows 11 x64 电脑，Intel Core i9-14900HX，16 GB 内存，NVIDIA GeForce RTX 5060 Laptop GPU（8 GB 显存）。三条路线分开运行。

| 路线 | 推理配置 |
| --- | --- |
| Qwen3-ASR 1.7B | YuYbUuRu v0.4.5，ONNX Runtime CPU 编码，8 个 intra-op 线程，llama.cpp Vulkan 解码。音频编码器为 FP16，解码器为 Q5_K_M。 |
| FireRedASR2-AED BF16 | 官方权重，PyTorch 2.7.1 + CUDA 12.8，BF16，beam=3。测试加载器在 Windows 使用内存映射读取 checkpoint 并将参数转入 GPU。纯识别关闭时间戳。 |
| FireRedASR2-AED INT8 | sherpa-onnx 1.13.8 的 AED v2 INT8 导出，CPU 8 线程，greedy search。 |

版本来源见 [FireRedASR2S 官方代码](https://github.com/FireRedTeam/FireRedASR2S/tree/4e7d9aaf4482a47cec1724807026b9b151926eb5)、[官方 checkpoint](https://huggingface.co/FireRedTeam/FireRedASR2-AED/tree/2304afed56eacfee6256dee5937ed22ffa0b64ec) 和 [sherpa-onnx AED v2 INT8 导出说明](https://k2-fsa.github.io/sherpa/onnx/FireRedAsr/pretrained.html#sherpa-onnx-fire-red-asr2-zh-en-int8-2026-02-26)。

音频统一为 16 kHz、单声道、PCM16。短段来自官方示例。长段使用下表六段公开音频，每段出现两次，片段之间插入 0.25 秒静音。

| 编号 | 来源 |
| --- | --- |
| A | [你好世界](https://github.com/FireRedTeam/FireRedASR2S/blob/4e7d9aaf4482a47cec1724807026b9b151926eb5/assets/hello_zh.wav) |
| B | [Hello speech](https://github.com/FireRedTeam/FireRedASR2S/blob/4e7d9aaf4482a47cec1724807026b9b151926eb5/assets/hello_en.wav) |
| C | [AISHELL 普通话](https://github.com/FireRedTeam/FireRedASR/blob/834635e4cf277ed8ca92049fc375b17c3dc20748/examples/wav/BAC009S0764W0121.wav) |
| D | [换一首歌](https://github.com/FireRedTeam/FireRedASR/blob/834635e4cf277ed8ca92049fc375b17c3dc20748/examples/wav/IT0011W0001.wav) |
| E | [对话片段](https://github.com/FireRedTeam/FireRedASR/blob/834635e4cf277ed8ca92049fc375b17c3dc20748/examples/wav/TEST_NET_Y0000000000_-KTKHdZ2fb8_S00000.wav) |
| F | [会议片段](https://github.com/FireRedTeam/FireRedASR/blob/834635e4cf277ed8ca92049fc375b17c3dc20748/examples/wav/TEST_MEETING_T0000000001_S00000.wav) |

长段顺序为 F、B、C、E、A、D、C、B、F、D、A、E，共 52.599875 秒。

冷启动每条路线启动 3 个独立进程，保留 Windows 文件缓存。热态按单段提交到返回完整文本计时。Qwen 和 INT8 分别在 3 个进程中测量各段，BF16 在同一个已加载进程中对每段测量 3 次。冷启动计入各自进程的初始化和加载流程，Python 导入及音频准备的计时边界因路线而异。短句数据对应识别请求，实际桌面录音、按键和粘贴需要另测。

## 转写耗时

| 测项，3 次中位数 | Qwen 1.7B | FireRed BF16 CUDA | FireRed INT8 CPU |
| --- | ---: | ---: | ---: |
| 新进程启动到模型就绪 | 3.546 秒 | 4.387 秒 | 2.973 秒 |
| 新进程启动到首条 2.32 秒中文完整文本 | 4.404 秒 | 5.071 秒 | 3.805 秒 |
| 热态 2.32 秒中文（A） | 0.409 秒 | 0.085 秒 | 0.815 秒 |
| 热态 2.24 秒英文（B） | 0.415 秒 | 0.114 秒 | 0.867 秒 |
| 热态 4.20 秒中文（C） | 0.565 秒 | 0.210 秒 | 1.686 秒 |
| 热态 52.60 秒中英混合 | 3.709 秒 | 5.289 秒 | 61.140 秒 |
| 长段实时因子（耗时除以音频时长） | 0.071 | 0.101 | 1.162 |

52.60 秒段三次用时范围分别为 Qwen 3.36–5.44 秒、BF16 5.28–5.31 秒、INT8 57.56–62.88 秒。按中位数计算，BF16 的三条热态短句耗时比 Qwen 低约 63%–79%，长段用时高约 43%。INT8 的长段中位用时约为 Qwen 的 16.5 倍。

另测了 Qwen 的文件转录命令，计入冷启动、音频读取、时间戳对齐和 TXT、JSON、SRT 导出。4.20 秒和 52.60 秒音频的完成用时中位数分别为 **4.277 秒**和 **11.122 秒**。FireRed 在本次使用独立纯识别流程，因此这组文件导出数据单列。

## 内存与显存

| 纯识别测试过程峰值，MiB | Qwen 1.7B | FireRed BF16 CUDA | FireRed INT8 CPU |
| --- | ---: | ---: | ---: |
| 总工作集 | 2,873 | 5,229 | 2,671 |
| 私有工作集 | 1,995 | 1,487 | 2,640 |
| 私有提交量 | 3,548 | 8,953 | 3,805 |
| 专用显存 | 1,491 | 3,231 | CPU 路径，未采样 |

Windows 主机内存按测试进程树采样，间隔约 20–25 ms。总工作集包含驻留的共享页面，私有工作集只统计私有驻留页面，私有提交量统计进程的私有内存承诺。GPU 数值来自 WDDM 进程计数器采样。表中各项可在不同时间达到峰值。

BF16 总工作集峰值出现在冷加载阶段。主测试中，52.60 秒请求返回后的总工作集采样约为 1,573 MiB。以全过程峰值计算，BF16 相对 Qwen 的总工作集约为 1.82 倍、私有提交量约为 2.52 倍、专用显存约为 2.17 倍。INT8 总工作集约少 203 MiB，私有提交量约多 258 MiB，差值按未取整的原值计算。

## 样本识别差异与选型

对输出与参考文本做 Unicode NFKC 和大小写归一化，仅保留字母和数字字符，再计算字符编辑距离。三条短句在所有路线下均匹配参考。52.60 秒长段共有 208 个归一化参考字符，Qwen、BF16 和 INT8 分别出现 8、6、4 处字符编辑差异。这些数字来自少量公开样本，其中长段为拼接语料，适用范围限于这组输入。

本次保留 Qwen 1.7B。BF16 的收益集中在连续短句，冷启动首条、这段长音频和资源峰值均未取得优势。INT8 可以走 CPU 推理，但此轮长段用时超过音频本身的时长。
