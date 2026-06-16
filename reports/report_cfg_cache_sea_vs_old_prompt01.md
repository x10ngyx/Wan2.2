# SeaCache 风格 CFG Cache 与原 CFG Cache 在 Ali Prompt 1 上的表现对比

## 1. 实验目的

本实验比较两种 CFG cache 方法在 Ali Prompt 1 上的速度和质量表现：

- 原 CFG cache：`--cfg_cache threshold`，使用原先的 CFG delta threshold 逻辑。
- SeaCache 风格 CFG cache：`--cfg_cache sea-threshold`，使用与 SeaCache 对齐的特征和 scheduler-aware SEA 过滤逻辑。

本实验只启用 CFG cache，关闭 timestep cache 和 block cache，用于观察 CFG cache 本身的质量/速度特性。

比较指标包括：

- 推理加速效果：使用 compute-only inference elapsed time 计算 speedup。
- 视频质量：使用同 prompt、同 seed、同 shape 的 no-cache baseline，通过 FFmpeg PSNR 计算平均 PSNR。
- 缓存行为：记录 CFG reuse / recompute 次数。

## 2. 实验配置

### 2.1 通用生成配置

| 项目 | 配置 |
|---|---|
| 模型任务 | Wan2.2 `t2v-A14B` |
| 模型权重 | `/hy-tmp/models/Wan2.2-T2V-A14B` |
| Prompt | Ali Prompt 1: A panda in a small red jacket and tiny hat plays a miniature acoustic guitar in a serene bamboo forest. |
| Prompt 来源 | `/hy-tmp/work/Wan2.2/prompt.txt` 的第 1 条 |
| Seed | `42` |
| 分辨率 | `832*480` |
| 帧数 | `45` |
| 采样步数 | `50` |
| 采样器 | `dpm++` |
| `sample_shift` | `12.0` |
| `sample_guide_scale` | `[3.0, 4.0]` |
| 推理参数 | `--offload_model --convert_model_dtype` |
| Timestep cache | `none` |
| Block cache | `none` |
| 速度指标 | `inference_compute_elapsed_seconds`，即 compute-only 推理时间 |
| 质量指标 | FFmpeg PSNR，对齐同 prompt/seed/shape 的 no-cache baseline |

Baseline 使用 `/hy-tmp/wan22_cache_ablation_prompt01_50step_45f_480p_20260609_184625` 中的 no-cache prompt 01 输出，baseline compute time 为 `522.603s`。

### 2.2 方法配置

| 方法 | CLI 配置 | 实验 threshold |
|---|---|---|
| 原 CFG cache | `--cfg_cache threshold --cfg_threshold <threshold>` | `0.02`, `0.03` |
| SeaCache 风格 CFG cache | `--cfg_cache sea-threshold --cfg_threshold <threshold>` | `0.10`, `0.20`, `0.30` |

原 CFG cache 的主要配置：

| 参数 | 值 |
|---|---|
| CFG metric | `timestep_modulated_input_rel_l1` |
| `cfg_start` / `cfg_end` | `0.1` / `0.9` |
| `cfg_max_reuse` | `3` |
| `cfg_eps` | `1e-06` |

SeaCache 风格 CFG cache 的主要配置：

| 参数 | 值 |
|---|---|
| `cfg_sea_power_exp` | `3.0` |
| `cfg_sea_power_const` | `1.0` |
| `cfg_sea_norm_mode` | `mean` |
| `cfg_ret_steps` | `1` |
| `cfg_cutoff_steps` | `1` |

说明：原 CFG cache 和 SeaCache 风格 CFG cache 的 threshold 不在同一指标空间中，不能直接比较 threshold 数值大小，只能比较最终 speedup、PSNR 和 CFG reuse/recompute 行为。

### 2.3 数据来源

| 数据 | 归档目录 |
|---|---|
| CFG cache 对比实验 | `/hy-tmp/wan22_cfg_cache_prompt01_50step_45f_480p_20260613_163243` |
| baseline 来源 | `/hy-tmp/wan22_cache_ablation_prompt01_50step_45f_480p_20260609_184625` |
| 结果表 | `/hy-tmp/wan22_cfg_cache_prompt01_50step_45f_480p_20260613_163243/results/summary_with_cache.csv` |

该实验共包含 5 个候选，全部完成，均有视频、ffprobe JSON、PSNR JSON、日志和命令记录。

## 3. 实验结果

### 3.1 主结果表

| 方法 | Threshold | 推理时间 (s) | Speedup | PSNR (dB) | Min PSNR (dB) | CFG Reuse / Recompute |
|---|---:|---:|---:|---:|---:|---:|
| 原 CFG `threshold` | 0.02 | 502.137 | 1.041x | 26.732 | 22.89 | 9 / 41 |
| 原 CFG `threshold` | 0.03 | 459.577 | 1.137x | 21.571 | 20.31 | 17 / 33 |
| Sea CFG `sea-threshold` | 0.10 | 518.870 | 1.007x | 37.457 | 34.81 | 6 / 44 |
| Sea CFG `sea-threshold` | 0.20 | 444.599 | 1.175x | 26.226 | 23.13 | 20 / 30 |
| Sea CFG `sea-threshold` | 0.30 | 402.930 | 1.297x | 21.359 | 20.07 | 28 / 22 |

### 3.2 分 stage 的 CFG 复用情况

| 方法 | Threshold | High-stage Reuse / Recompute | Low-stage Reuse / Recompute |
|---|---:|---:|---:|
| 原 CFG `threshold` | 0.02 | 7 / 25 | 2 / 16 |
| 原 CFG `threshold` | 0.03 | 13 / 19 | 4 / 14 |
| Sea CFG `sea-threshold` | 0.10 | 0 / 32 | 6 / 12 |
| Sea CFG `sea-threshold` | 0.20 | 10 / 22 | 10 / 8 |
| Sea CFG `sea-threshold` | 0.30 | 16 / 16 | 12 / 6 |

## 4. 简单结论

1. SeaCache 风格 CFG cache 的质量/速度曲线比原 CFG cache 更有弹性。Sea CFG `0.10` 是保守高质量点，PSNR 达到 `37.457 dB`，但几乎没有加速，speedup 只有 `1.007x`。

2. 在中等质量区间，Sea CFG `0.20` 与原 CFG `0.02` 的 PSNR 接近，分别为 `26.226 dB` 和 `26.732 dB`；但 Sea CFG `0.20` 的速度更好，达到 `1.175x`，高于原 CFG `0.02` 的 `1.041x`。

3. 在更激进的设置下，Sea CFG `0.30` 与原 CFG `0.03` 的 PSNR 接近，分别为 `21.359 dB` 和 `21.571 dB`；但 Sea CFG `0.30` 的 speedup 为 `1.297x`，明显高于原 CFG `0.03` 的 `1.137x`。

4. 从复用行为看，Sea CFG 随 threshold 增大能更有效地增加 CFG reuse 次数：从 `6/44` 增加到 `20/30`、`28/22`。原 CFG cache 从 `0.02` 到 `0.03` 也能增加复用，但在相近质量下速度收益不如 Sea CFG。

总体看，在 Ali Prompt 1 的 CFG-only 对比中，SeaCache 风格 CFG cache 优于原 CFG cache：保守阈值能给出更高质量，中等和激进阈值能在接近原方法质量的情况下取得更高加速。
