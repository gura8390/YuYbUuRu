# CUDA 技术路线归档

本轮实验完成于 2026 年 10 月 27 日，归档于 2026 年 11 月 6 日。实验实现并验证了 Windows 上的 CUDA 解码和显存优化。正式路线采用 CPU 编码与 Vulkan 解码，CUDA 实验分支转为历史查阅用途。

[历史分支 0.4.3—CUDA](https://github.com/gura8390/YuYbUuRu/tree/0.4.3%E2%80%94CUDA) 保留源码和构建说明。CUDA 候选版本为 0.4.3-cuda.1，最后一次实验提交是 [3b7c123](https://github.com/gura8390/YuYbUuRu/commit/3b7c123df6fe8cb8e0c220f65258652297419fa5)，基线为 [Vulkan v0.4.3 的 73bd065](https://github.com/gura8390/YuYbUuRu/commit/73bd065b1453c8496e08dec054bf745ad5b15bfa)。

## 做过的验证

- 在同一台 Windows x64 电脑上测试 i9-14900HX 和 RTX 5060 Laptop 8 GB 的 CUDA 12.8 与 Vulkan 路线。比较使用 Qwen3-ASR 0.6B，保留 CPU 音频编码器和 CPU 文件对齐。
- 实现 ASR 单输出预留，限制实际只需一行输出的推理请求所用的 logits 缓冲。CUDA 候选另外使用 cuBLAS 延迟加载，并验证实际 fallback 的已知矩阵结果。
- 评估了 ORT CUDA 编码、动态 ASR 上下文和实验性的多流 CUDA Graph。多流方案曾在混合长度回归中出现一次内核启动失败，最终候选保留普通 CUDA Graph。

原生改动可查阅 [ASR 单输出预留](https://github.com/gura8390/YuYbUuRu/blob/3b7c123df6fe8cb8e0c220f65258652297419fa5/native/asr-single-output.md)、[cuBLAS 延迟加载](https://github.com/gura8390/YuYbUuRu/blob/3b7c123df6fe8cb8e0c220f65258652297419fa5/native/cuda-delayload.md) 和 [ORT CUDA 编码器实验](https://github.com/gura8390/YuYbUuRu/blob/3b7c123df6fe8cb8e0c220f65258652297419fa5/native/ort-cuda-encoder.md)。

## 实测结果

以下为四组各 3 轮、每轮 39 个固定识别请求的中位数。整轮耗时包含完整请求链路。显存是 WDDM 进程计数器在阶段点的观测值，主机私有提交量和显存分别统计。

| 路线 | 主机私有提交峰值（MiB） | 整轮耗时（秒） | 专用显存阶段峰值（MiB） |
| --- | ---: | ---: | ---: |
| Vulkan 0.4.3 原配置 | 1,482.73 | 16.938 | 918.82 |
| CUDA 未优化基线 | 1,940.51 | 16.113 | 1,023.49 |
| CUDA 单输出预留 + cuBLAS 延迟加载 | 1,431.93 | 17.789 | 749.48 |
| Vulkan 同样启用单输出预留 | 1,209.85 | 17.433 | 647.60 |

CUDA 候选相对其基线降低了 **26.21%** 的私有提交峰值。固定同一 prompt 和 255 个外供 token 的解码回放中，七次运行的后六次热态中位数为 CUDA **324.57 token/s**，Vulkan **279.80 token/s**。在同样启用单输出预留的完整请求对照中，Vulkan 的整轮时间少 0.356 秒，私有提交少 222.08 MiB，专用显存阶段峰值少 101.88 MiB。

最终两条路线各自通过 **117/117** 个识别请求和 **7/7** 个文件转录请求的逐项对照。CUDA 与 Vulkan 在 39 条固定输入上有 11 条输出不同，该组对照缺少人工真值，差异计数只用于回归分析。文件测试走本地转录链路，实际桌面按键、录音和粘贴属于另外的验收范围。

## 归档决定

纯解码回放显示了 CUDA 的吞吐潜力。完整识别链路仍包含 CPU 音频编码、采样与进程通信。本轮使用相同的单输出预留后，Vulkan 在整体用时、主机私有提交量和专用显存阶段观测上都占优。因此继续维护 Vulkan 主线，CUDA 代码与这份结果保留供查阅。
