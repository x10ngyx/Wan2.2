# SeaCache 与 ZEUS-threshold 在 Ali Prompt 1/2 上的表现对比

## 1. 实验目的

本实验比较两种 timestep 级缓存方法在 Ali 测试集前两个 prompt 上的速度和质量表现：

- `ZEUS-threshold`：基于 latent relative-L1 的 threshold timestep cache。
- `SeaCache`：基于 SeaCache 指标的 timestep/residual cache 方法。

比较指标包括：

- 推理加速效果：使用 compute-only inference elapsed time 计算 speedup。
- 视频质量：使用同 prompt、同 seed、同 shape 的 no-cache baseline，通过 FFmpeg PSNR 计算平均 PSNR。
- 缓存行为：记录 timestep reuse / recompute 次数。

本报告只做简单结论分析，重点保留实验配置与结果数据。

## 2. 实验配置

### 2.1 通用生成配置

| 项目 | 配置 |
|---|---|
| 模型任务 | Wan2.2 `t2v-A14B` |
| 模型权重 | `/hy-tmp/models/Wan2.2-T2V-A14B` |
| Prompt 集 | `test_sets/ali_10/prompts.txt` 的前两个 prompt |
| Prompt 1 summary | A panda in a small red jacket and tiny hat plays a miniature acoustic guitar in a serene bamboo forest. |
| Prompt 2 summary | Summer beach vacation style: a white cat wearing sunglasses sits on a surfboard. |
| Seed | `42` |
| 分辨率 | `832*480` |
| 帧数 | `45` |
| 采样步数 | `50` |
| 采样器 | `dpm++` |
| 推理参数 | `--offload_model --convert_model_dtype` |
| 速度指标 | `inference_compute_elapsed_seconds`，即 compute-only 推理时间 |
| 质量指标 | FFmpeg PSNR，对齐同 prompt/seed/shape 的 no-cache baseline |
| PSNR 统计 | 排除 perfect / Infinity frames 后汇总；表中 `Infinity` 表示该候选与 baseline 完全一致或无有效有限误差帧 |

### 2.2 方法配置

| 方法 | CLI 配置 | threshold 范围 |
|---|---|---|
| ZEUS-threshold | `--timestep_cache zeus-threshold --timestep_threshold <threshold>` | `0.005`, `0.020`, `0.080`, `0.200`, `0.600` |
| SeaCache | `--timestep_cache seacache --seacache_threshold <threshold>` | Prompt 1: `0.050`, `0.100`, `0.200`, `0.300`, `0.500`; Prompt 2: `0.080`, `0.100`, `0.120`, `0.150`, `0.180`, `0.200`, `0.250`, `0.300`, `0.400`, `0.500` |

说明：SeaCache 与 ZEUS-threshold 的 threshold 数值不在同一指标空间中，不能直接按数值大小比较，只能按最终速度、PSNR 和复用行为比较。

### 2.3 数据来源

| 数据 | 归档目录 |
|---|---|
| ZEUS-threshold prompt 1/2 | `/hy-tmp/wan22_zeus_threshold_reuse_interp_10prompt_5th_20260608_195427` |
| SeaCache prompt 1 | `/hy-tmp/wan22_seacache_50step_45f_480p_20260611_191733` |
| SeaCache prompt 2 dense sweep | `/hy-tmp/wan22_seacache_prompt02_dense_20260611_204826` |
| SeaCache prompt 2 high-threshold sweep | `/hy-tmp/wan22_seacache_prompt02_highthr_20260612_000218` |

## 3. 实验结果

### 3.1 Ali Prompt 1

Prompt 1 的 no-cache baseline compute time 为 `522.603s`。

| 方法 | Threshold | 推理时间 (s) | Speedup | PSNR (dB) | Reuse / Recompute |
|---|---:|---:|---:|---:|---:|
| ZEUS-threshold | 0.005 | 470.287 | 1.111x | 26.954 | 5 / 45 |
| ZEUS-threshold | 0.020 | 325.253 | 1.607x | 18.606 | 19 / 31 |
| ZEUS-threshold | 0.080 | 232.145 | 2.251x | 18.873 | 28 / 22 |
| ZEUS-threshold | 0.200 | 201.093 | 2.599x | 18.900 | 31 / 19 |
| ZEUS-threshold | 0.600 | 191.124 | 2.734x | 18.928 | 32 / 18 |
| SeaCache | 0.050 | 529.025 | 0.988x | Infinity | 0 / 50 |
| SeaCache | 0.100 | 469.995 | 1.112x | 36.303 | 6 / 44 |
| SeaCache | 0.200 | 333.084 | 1.569x | 24.558 | 20 / 30 |
| SeaCache | 0.300 | 265.758 | 1.966x | 20.562 | 27 / 23 |
| SeaCache | 0.500 | 188.053 | 2.779x | 19.460 | 35 / 15 |

### 3.2 Ali Prompt 2

Prompt 2 的 no-cache baseline compute time 为 `522.608s`。

| 方法 | Threshold | 推理时间 (s) | Speedup | PSNR (dB) | Reuse / Recompute |
|---|---:|---:|---:|---:|---:|
| ZEUS-threshold | 0.005 | 478.997 | 1.091x | 24.433 | 4 / 46 |
| ZEUS-threshold | 0.020 | 344.151 | 1.519x | 20.112 | 17 / 33 |
| ZEUS-threshold | 0.080 | 230.705 | 2.265x | 20.433 | 28 / 22 |
| ZEUS-threshold | 0.200 | 199.380 | 2.621x | 20.432 | 31 / 19 |
| ZEUS-threshold | 0.600 | 188.933 | 2.766x | 20.438 | 32 / 18 |
| SeaCache | 0.080 | 528.725 | 0.988x | Infinity | 0 / 50 |
| SeaCache | 0.100 | 479.491 | 1.090x | 45.532 | 5 / 45 |
| SeaCache | 0.120 | 421.440 | 1.240x | 42.475 | 11 / 39 |
| SeaCache | 0.150 | 372.696 | 1.402x | 35.441 | 16 / 34 |
| SeaCache | 0.180 | 343.774 | 1.520x | 29.848 | 19 / 31 |
| SeaCache | 0.200 | 334.663 | 1.562x | 30.097 | 20 / 30 |
| SeaCache | 0.250 | 285.500 | 1.831x | 29.055 | 25 / 25 |
| SeaCache | 0.300 | 265.988 | 1.965x | 29.582 | 27 / 23 |
| SeaCache | 0.400 | 217.301 | 2.405x | 27.044 | 32 / 18 |
| SeaCache | 0.500 | 197.847 | 2.641x | 23.725 | 34 / 16 |

## 4. 简单结论

1. SeaCache 在这两个 Ali prompt 上整体优于 ZEUS-threshold，尤其是在相近 speedup 下能保持更高 PSNR。

2. 在 Prompt 1 上，SeaCache `0.200` 与 ZEUS-threshold `0.020` 的速度接近，分别为 `1.569x` 和 `1.607x`，但 SeaCache 的 PSNR 为 `24.558 dB`，明显高于 ZEUS-threshold 的 `18.606 dB`。在更激进设置下，SeaCache `0.500` 达到 `2.779x / 19.460 dB`，也略好于 ZEUS-threshold `0.600` 的 `2.734x / 18.928 dB`。

3. 在 Prompt 2 上，SeaCache 优势更明显。约 `1.5x` 加速时，SeaCache `0.180/0.200` 的 PSNR 约为 `29.85-30.10 dB`，而 ZEUS-threshold `0.020` 只有 `20.112 dB`。约 `2.4x` 加速时，SeaCache `0.400` 仍有 `27.044 dB`，ZEUS-threshold 在相近速度区间约为 `20.43 dB`。

4. ZEUS-threshold 的质量下降较早：当 threshold 从 `0.005` 增大到 `0.020` 后，PSNR 明显下滑，后续继续增大 threshold 时主要提升速度，但 PSNR 基本停留在较低区间。SeaCache 的质量-速度曲线更平滑，可用 threshold 区间更宽。

总体看，在 Ali Prompt 1/2 这组小规模对比中，SeaCache 是更合适的 timestep cache 基线；ZEUS-threshold 可保留为对照方法，但不应作为后续质量优先实验的主推荐方法。
