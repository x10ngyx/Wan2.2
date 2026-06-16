# SeaCache 风格 Block Cache 与原 Block-Group Cache 在 Ali Prompt 1 上的表现对比

## 1. 实验目的

本实验比较两种 block cache 方法在 Ali Prompt 1 上的速度和质量表现：

- 原 block-group cache：`--block_cache block-group`，使用 pooled relative-L1 指标和固定最大连续复用限制。
- SeaCache 风格 block-group cache：仍使用 `--block_cache block-group` 接口，但指标改为 `sea_full_rel_l1`，决策方式改为 `accumulated`，使用与 SeaCache 对齐的 full-feature / scheduler-aware 过滤特征。

本实验只启用 block cache，关闭 timestep cache 和 CFG cache，用于观察 block cache 本身的质量/速度特性。

比较指标包括：

- 推理加速效果：使用 compute-only inference elapsed time 计算 speedup。
- 视频质量：使用同 prompt、同 seed、同 shape 的 no-cache baseline，通过 FFmpeg PSNR 计算平均 PSNR。
- 缓存行为：记录 block-group reuse / recompute 次数。

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
| CFG cache | `none` |
| 速度指标 | `inference_compute_elapsed_seconds`，即 compute-only 推理时间 |
| 质量指标 | FFmpeg PSNR，对齐同 prompt/seed/shape 的 no-cache baseline |

Baseline 使用 `/hy-tmp/wan22_cache_ablation_prompt01_50step_45f_480p_20260609_184625` 中的 no-cache prompt 01 输出，baseline compute time 为 `522.603s`。

### 2.2 方法配置

| 方法 | CLI 配置 | 实验 threshold |
|---|---|---|
| 原 block-group cache | `--block_cache block-group --block_group_metric pooled_rel_l1 --block_group_threshold <threshold>` | `0.01`, `0.03`, `0.05` |
| SeaCache 风格 block-group cache | `--block_cache block-group --block_group_metric sea_full_rel_l1 --block_group_decision accumulated --block_group_threshold <threshold>` | `0.05`, `0.10`, `0.20` |

原 block-group cache 的主要配置：

| 参数 | 值 |
|---|---|
| `block_group_size` | `5` |
| `block_group_metric` | `pooled_rel_l1` |
| `block_group_start` / `block_group_end` | `0.1` / `0.9` |
| `block_group_max_reuse` | `3` |

SeaCache 风格 block-group cache 的主要配置：

| 参数 | 值 |
|---|---|
| `block_group_size` | `5` |
| `block_group_metric` | `sea_full_rel_l1` |
| `block_group_decision` | `accumulated` |
| `block_group_ret_steps` | `1` |
| `block_group_cutoff_steps` | `1` |
| `block_group_max_reuse` | `50` |
| `block_group_sea_power_exp` | `3.0` |
| `block_group_sea_power_const` | `1.0` |
| `block_group_sea_norm_mode` | `mean` |

说明：两种方法的 threshold 不在同一指标空间中，不能直接比较 threshold 数值大小，只能比较最终 speedup、PSNR 和 reuse/recompute 行为。SeaCache 风格 block-group cache 需要额外保存 full filtered features，显存压力高于原 block-group cache；本实验使用逐 threshold 独立 `generate.py` 子进程运行，以隔离潜在 OOM。

### 2.3 数据来源

| 数据 | 归档目录 |
|---|---|
| 原 block-group cache | `/hy-tmp/wan22_block_cache_only_50step_45f_480p_20260609_125436` |
| SeaCache 风格 block-group cache | `/hy-tmp/wan22_block_group_sea_full_prompt01_50step_45f_480p_20260614_235605` |
| baseline 来源 | `/hy-tmp/wan22_cache_ablation_prompt01_50step_45f_480p_20260609_184625` |
| 原 block-group 结果表 | `/hy-tmp/wan22_block_cache_only_50step_45f_480p_20260609_125436/results/aggregate_by_method_threshold.csv` |
| Sea block 结果表 | `/hy-tmp/wan22_block_group_sea_full_prompt01_50step_45f_480p_20260614_235605/results/summary.csv` |

补充说明：`/hy-tmp/wan22_block_group_sea_full_prompt01_50step_45f_480p_20260613_235449` 是早期失败 pilot，未产生有效结果行，不作为本报告结果来源。

## 3. 实验结果

### 3.1 主结果表

| 方法 | Threshold | 推理时间 (s) | Speedup | PSNR (dB) | Min PSNR (dB) | Reuse / Recompute |
|---|---:|---:|---:|---:|---:|---:|
| 原 block-group `pooled_rel_l1` | 0.01 | 532.389 | 0.982x | Infinity | Infinity | 0 / 800 |
| 原 block-group `pooled_rel_l1` | 0.03 | 384.116 | 1.361x | 19.396 | 17.80 | 244 / 556 |
| 原 block-group `pooled_rel_l1` | 0.05 | 305.079 | 1.713x | 19.491 | 18.04 | 374 / 426 |
| Sea block-group `sea_full_rel_l1` | 0.05 | 544.867 | 0.959x | Infinity | Infinity | 0 / 800 |
| Sea block-group `sea_full_rel_l1` | 0.10 | 506.218 | 1.032x | 37.465 | 35.93 | 68 / 732 |
| Sea block-group `sea_full_rel_l1` | 0.20 | 362.200 | 1.443x | 24.815 | 21.93 | 306 / 494 |

### 3.2 SeaCache 风格 block-group 的分 stage 复用情况

| Threshold | High-stage Reuse / Recompute | Low-stage Reuse / Recompute |
|---:|---:|---:|
| 0.05 | 0 / 512 | 0 / 288 |
| 0.10 | 0 / 512 | 68 / 220 |
| 0.20 | 160 / 352 | 146 / 142 |

说明：原 block-group 的历史结果表只保留了汇总 reuse/recompute 和详细路径字符串，本报告主表使用汇总值；Sea block 的 high/low 分布是从候选日志中的 `Block-group cache summary` 重新汇总得到。

## 4. 简单结论

1. SeaCache 风格 block-group cache 在质量保持上明显优于原 block-group cache。原 block-group 在 `1.36x-1.71x` 加速区间 PSNR 约为 `19.4 dB`；Sea block `0.20` 在 `1.443x` 时仍有 `24.815 dB`，质量高出约 `5.4 dB`。

2. Sea block `0.10` 是高质量保守点，达到 `37.465 dB`，但加速很小，只有 `1.032x`。对应地，原 block-group `0.01` 和 Sea block `0.05` 都几乎不复用，PSNR 为 `Infinity`，但速度低于 baseline，说明没有复用时 cache 判断本身会带来额外开销。

3. 原 block-group `0.05` 的速度更高，达到 `1.713x`，但 PSNR 只有 `19.491 dB`。Sea block `0.20` 速度较低，为 `1.443x`，但 PSNR 明显更好，适合质量优先的 block cache 方向。

4. 从复用行为看，Sea block 在 `0.10` 时主要复用 low stage，high stage 仍完全重算；到 `0.20` 时 high/low stage 都开始大量复用。这说明 Sea full accumulated 指标的复用增长更平滑，但也带来额外计算和显存开销。

总体看，在 Ali Prompt 1 的 block-cache-only 对比中，SeaCache 风格 block-group cache 更适合高质量或中等质量目标；原 block-group cache 可以给出更高速度，但质量下降更早。后续若要把 Sea block 纳入三缓存组合，需要继续关注 full-feature 缓存带来的显存压力和额外判断开销。
