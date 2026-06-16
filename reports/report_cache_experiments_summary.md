# Wan2.2 T2V Cache 实验总报告

## 1. 报告范围

本报告汇总四组已完成实验：

1. SeaCache 与 ZEUS-threshold 在 Ali Prompt 1/2 上的 timestep cache 对比。
2. SeaCache 风格 CFG cache 与原 CFG cache 的 CFG-only 对比。
3. SeaCache 风格 block cache 与原 group block cache 的 block-only 对比。
4. Sea-style timestep / block / CFG 三缓存 threshold 组合实验。

本报告保留实验配置和关键运行结果。附加统计只保留最必要的完成状态、最快点、最高质量点和推荐点。

## 2. 通用实验配置

| 项目 | 配置 |
|---|---|
| 模型任务 | Wan2.2 `t2v-A14B` |
| 模型权重 | `/hy-tmp/models/Wan2.2-T2V-A14B` |
| Seed | `42` |
| 分辨率 | `832*480` |
| 帧数 | `45` |
| 采样步数 | `50` |
| 采样器 | `dpm++` |
| 推理参数 | `--offload_model --convert_model_dtype` |
| 速度指标 | `inference_compute_elapsed_seconds`，即 compute-only 推理时间 |
| 质量指标 | FFmpeg PSNR，对齐同 prompt / seed / shape 的 no-cache baseline |
| PSNR 统计 | 排除 perfect / Infinity frames 后汇总；`Infinity` 表示与 baseline 等价或无有效有限误差帧 |

Prompt 信息：

| Prompt | Summary |
|---|---|
| Ali Prompt 1 | A panda in a small red jacket and tiny hat plays a miniature acoustic guitar in a serene bamboo forest. |
| Ali Prompt 2 | Summer beach vacation style: a white cat wearing sunglasses sits on a surfboard. |

主要 baseline：

| Prompt | Baseline 来源 | Baseline compute time |
|---|---|---:|
| Ali Prompt 1 | `/hy-tmp/wan22_cache_ablation_prompt01_50step_45f_480p_20260609_184625` | 522.603 s |
| Ali Prompt 2 | no-cache prompt 2 baseline | 522.608 s |

## 3. 实验一：SeaCache vs ZEUS-threshold

### 3.1 实验配置

| 项目 | 配置 |
|---|---|
| Prompt | Ali Prompt 1、Ali Prompt 2 |
| 实验目标 | 比较 timestep 级缓存方法的速度、质量和 reuse/recompute 行为 |
| ZEUS-threshold CLI | `--timestep_cache zeus-threshold --timestep_threshold <threshold>` |
| ZEUS-threshold thresholds | `0.005`, `0.020`, `0.080`, `0.200`, `0.600` |
| SeaCache CLI | `--timestep_cache seacache --seacache_threshold <threshold>` |
| SeaCache Prompt 1 thresholds | `0.050`, `0.100`, `0.200`, `0.300`, `0.500` |
| SeaCache Prompt 2 thresholds | `0.080`, `0.100`, `0.120`, `0.150`, `0.180`, `0.200`, `0.250`, `0.300`, `0.400`, `0.500` |
| ZEUS 数据源 | `/hy-tmp/wan22_zeus_threshold_reuse_interp_10prompt_5th_20260608_195427` |
| SeaCache Prompt 1 数据源 | `/hy-tmp/wan22_seacache_50step_45f_480p_20260611_191733` |
| SeaCache Prompt 2 数据源 | `/hy-tmp/wan22_seacache_prompt02_dense_20260611_204826`, `/hy-tmp/wan22_seacache_prompt02_highthr_20260612_000218` |

说明：ZEUS-threshold 和 SeaCache 的 threshold 不在同一指标空间，只比较最终 speedup、PSNR 和 reuse/recompute。

### 3.2 Ali Prompt 1 完整结果

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

### 3.3 Ali Prompt 2 完整结果

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

### 3.4 结论

SeaCache 在相近 speedup 下 PSNR 明显高于 ZEUS-threshold，是更合适的 timestep cache 方法。

## 4. 实验二：SeaCache 风格 CFG Cache vs 原 CFG Cache

### 4.1 实验配置

| 项目 | 配置 |
|---|---|
| Prompt | Ali Prompt 1 |
| 实验目标 | CFG-only，对比 CFG cache 本身的速度、质量和 reuse/recompute 行为 |
| Timestep cache | `none` |
| Block cache | `none` |
| `sample_shift` | `12.0` |
| `sample_guide_scale` | `[3.0, 4.0]` |
| 原 CFG CLI | `--cfg_cache threshold --cfg_threshold <threshold>` |
| 原 CFG thresholds | `0.02`, `0.03` |
| 原 CFG metric | `timestep_modulated_input_rel_l1` |
| 原 CFG `cfg_start` / `cfg_end` | `0.1` / `0.9` |
| 原 CFG `cfg_max_reuse` | `3` |
| Sea CFG CLI | `--cfg_cache sea-threshold --cfg_threshold <threshold>` |
| Sea CFG thresholds | `0.10`, `0.20`, `0.30` |
| Sea CFG `cfg_ret_steps` / `cfg_cutoff_steps` | `1` / `1` |
| Sea CFG SEA 参数 | `cfg_sea_power_exp=3.0`, `cfg_sea_power_const=1.0`, `cfg_sea_norm_mode=mean` |
| 数据源 | `/hy-tmp/wan22_cfg_cache_prompt01_50step_45f_480p_20260613_163243` |
| 结果表 | `/hy-tmp/wan22_cfg_cache_prompt01_50step_45f_480p_20260613_163243/results/summary_with_cache.csv` |
| 完成状态 | 5/5 候选完成 |

### 4.2 完整结果

| 方法 | Threshold | 推理时间 (s) | Speedup | PSNR (dB) | Min PSNR (dB) | CFG Reuse / Recompute | High-stage Reuse / Recompute | Low-stage Reuse / Recompute |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 原 CFG `threshold` | 0.02 | 502.137 | 1.041x | 26.732 | 22.89 | 9 / 41 | 7 / 25 | 2 / 16 |
| 原 CFG `threshold` | 0.03 | 459.577 | 1.137x | 21.571 | 20.31 | 17 / 33 | 13 / 19 | 4 / 14 |
| Sea CFG `sea-threshold` | 0.10 | 518.870 | 1.007x | 37.457 | 34.81 | 6 / 44 | 0 / 32 | 6 / 12 |
| Sea CFG `sea-threshold` | 0.20 | 444.599 | 1.175x | 26.226 | 23.13 | 20 / 30 | 10 / 22 | 10 / 8 |
| Sea CFG `sea-threshold` | 0.30 | 402.930 | 1.297x | 21.359 | 20.07 | 28 / 22 | 16 / 16 | 12 / 6 |

### 4.3 结论

Sea CFG 在相近 PSNR 下速度更高；`0.20` 和 `0.30` 分别优于原 CFG `0.02` 和 `0.03` 的速度表现。

## 5. 实验三：SeaCache 风格 Block Cache vs 原 Group Block Cache

### 5.1 实验配置

| 项目 | 配置 |
|---|---|
| Prompt | Ali Prompt 1 |
| 实验目标 | Block-only，对比 block cache 本身的速度、质量和 reuse/recompute 行为 |
| Timestep cache | `none` |
| CFG cache | `none` |
| `sample_shift` | `12.0` |
| `sample_guide_scale` | `[3.0, 4.0]` |
| 原 block CLI | `--block_cache block-group --block_group_metric pooled_rel_l1 --block_group_threshold <threshold>` |
| 原 block thresholds | `0.01`, `0.03`, `0.05` |
| 原 block 主要参数 | `block_group_size=5`, `block_group_start=0.1`, `block_group_end=0.9`, `block_group_max_reuse=3` |
| Sea block CLI | `--block_cache block-group --block_group_metric sea_full_rel_l1 --block_group_decision accumulated --block_group_threshold <threshold>` |
| Sea block thresholds | `0.05`, `0.10`, `0.20` |
| Sea block 主要参数 | `block_group_size=5`, `block_group_max_reuse=50`, `block_group_ret_steps=1`, `block_group_cutoff_steps=1` |
| Sea block SEA 参数 | `block_group_sea_power_exp=3.0`, `block_group_sea_power_const=1.0`, `block_group_sea_norm_mode=mean` |
| 原 block 数据源 | `/hy-tmp/wan22_block_cache_only_50step_45f_480p_20260609_125436` |
| Sea block 数据源 | `/hy-tmp/wan22_block_group_sea_full_prompt01_50step_45f_480p_20260614_235605` |
| 失败 pilot | `/hy-tmp/wan22_block_group_sea_full_prompt01_50step_45f_480p_20260613_235449`，不作为结果来源 |

说明：Sea block 使用 full filtered features，显存压力高于原 block-group；本实验按 threshold 独立子进程运行以隔离潜在 OOM。

### 5.2 完整结果

| 方法 | Threshold | 推理时间 (s) | Speedup | PSNR (dB) | Min PSNR (dB) | Reuse / Recompute | High-stage Reuse / Recompute | Low-stage Reuse / Recompute |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 原 block-group `pooled_rel_l1` | 0.01 | 532.389 | 0.982x | Infinity | Infinity | 0 / 800 | - | - |
| 原 block-group `pooled_rel_l1` | 0.03 | 384.116 | 1.361x | 19.396 | 17.80 | 244 / 556 | - | - |
| 原 block-group `pooled_rel_l1` | 0.05 | 305.079 | 1.713x | 19.491 | 18.04 | 374 / 426 | - | - |
| Sea block-group `sea_full_rel_l1` | 0.05 | 544.867 | 0.959x | Infinity | Infinity | 0 / 800 | 0 / 512 | 0 / 288 |
| Sea block-group `sea_full_rel_l1` | 0.10 | 506.218 | 1.032x | 37.465 | 35.93 | 68 / 732 | 0 / 512 | 68 / 220 |
| Sea block-group `sea_full_rel_l1` | 0.20 | 362.200 | 1.443x | 24.815 | 21.93 | 306 / 494 | 160 / 352 | 146 / 142 |

### 5.3 结论

Sea block 的质量保持明显更好；原 block 速度更高但质量更早降到约 `19.4 dB`。

## 6. 实验四：Sea-Style 三缓存 Threshold 组合

### 6.1 实验配置

| 项目 | 配置 |
|---|---|
| Prompt | Ali Prompt 1 |
| 实验目标 | 扫描 timestep / block / CFG 三类 Sea-style cache 的 threshold 组合 |
| Cache 顺序 | CFG cache 最外层；实际运行的 branch 先判断 timestep cache；timestep miss 后进入 block cache；两者都 miss 才执行 transformer blocks |
| Timestep cache | `--timestep_cache seacache --seacache_threshold <threshold>` |
| Timestep thresholds | `0.05`, `0.10`, `0.20`, `0.40`, `1.00` |
| Timestep 主要参数 | `seacache_power_exp=3.0`, `seacache_power_const=1.0`, `seacache_norm_mode=mean`, `seacache_use_ret_steps=false` |
| Block cache | `--block_cache block-group --block_group_metric sea_full_rel_l1 --block_group_decision accumulated --block_group_threshold <threshold>` |
| Block thresholds | `0.05`, `0.10`, `0.20`, `0.40`, `1.00` |
| Block 主要参数 | `block_group_size=5`, `block_group_max_reuse=50`, `block_group_ret_steps=1`, `block_group_cutoff_steps=1` |
| Block SEA 参数 | `block_group_sea_power_exp=3.0`, `block_group_sea_power_const=1.0`, `block_group_sea_norm_mode=mean` |
| CFG cache | `--cfg_cache sea-threshold --cfg_threshold <threshold>` |
| CFG thresholds | `0.05`, `0.10`, `0.20`, `0.40`, `1.00` |
| CFG 主要参数 | `cfg_ret_steps=1`, `cfg_cutoff_steps=1`, `cfg_sea_power_exp=3.0`, `cfg_sea_power_const=1.0`, `cfg_sea_norm_mode=mean` |
| 组合数 | `5 * 5 * 5 = 125` |
| 数据源 | `/hy-tmp/wan22_three_cache_sea_prompt01_50step_45f_480p_20260614_005404` |
| 结果表 | `/hy-tmp/wan22_three_cache_sea_prompt01_50step_45f_480p_20260614_005404/results/summary.csv` |
| 完成状态 | 125/125 候选完成，`failed/` 为空 |
| ffprobe | 所有候选均为 `832x480`、`45` 帧、`16 fps`、时长 `2.812500s` |

### 6.2 最关键结果

| 项目 | Candidate | Time (s) | Speedup | PSNR (dB) | Min PSNR (dB) | Reuse: timestep / block / CFG |
|---|---|---:|---:|---:|---:|---:|
| 无复用等价 baseline | `sea_ts_0p05__sea_bg_0p05__sea_cfg_0p05` | 569.448 | 0.918x | Infinity | Infinity | 0 / 0 / 0 |
| 最高有限 PSNR | `sea_ts_0p05__sea_bg_0p10__sea_cfg_0p05` | 529.685 | 0.987x | 37.465 | 35.93 | 0 / 68 / 0 |
| 高质量最快点，PSNR >= 35 dB | `sea_ts_0p10__sea_bg_0p10__sea_cfg_0p10` | 509.720 | 1.025x | 36.747 | 35.04 | 6 / 0 / 6 |
| PSNR >= 26 dB 最快点 | `sea_ts_0p10__sea_bg_0p05__sea_cfg_0p20` | 432.758 | 1.208x | 26.430 | 23.14 | 6 / 0 / 20 |
| PSNR >= 24 dB 最快点 | `sea_ts_0p20__sea_bg_0p20__sea_cfg_0p20` | 349.338 | 1.496x | 24.898 | 22.12 | 22 / 2 / 20 |
| PSNR >= 19 dB 最快点 | `sea_ts_0p40__sea_bg_0p10__sea_cfg_1p00` | 183.661 | 2.845x | 19.007 | 18.05 | 32 / 0 / 41 |
| PSNR >= 18 dB 最快点 | `sea_ts_1p00__sea_bg_0p05__sea_cfg_0p20` | 146.186 | 3.575x | 18.233 | 17.35 | 42 / 0 / 20 |
| 最快点 | `sea_ts_1p00__sea_bg_1p00__sea_cfg_1p00` | 92.597 | 5.644x | 11.914 | 11.66 | 42 / 40 / 41 |

### 6.3 结论

高质量区加速有限；`0.20/0.20/0.20` 是当前最清晰的中等质量平衡点，达到 `1.496x / 24.898 dB`。

## 7. 总结论

SeaCache 风格方法整体优于原 ZEUS / CFG / block 对照方法。高质量目标下加速空间有限；若接受约 `24-25 dB`，三缓存组合可达到约 `1.5x`；更高速度会快速牺牲质量。

## 附录：Sea-Style 三缓存 125 组完整结果表

下表为 `125` 个候选的完整结果。为便于阅读，只保留 threshold、速度、PSNR 和三类 cache 的 reuse/recompute 统计；完整路径与详细 trace 见归档目录中的 `results/summary.csv`。

| Candidate | TS | Block | CFG | Time (s) | Speedup | PSNR (dB) | Min PSNR (dB) | TS Reuse/Recompute | Block Reuse/Recompute | CFG Reuse/Recompute |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `sea_ts_0p05__sea_bg_0p05__sea_cfg_0p05` | 0.05 | 0.05 | 0.05 | 569.448 | 0.918x | Infinity | Infinity | 0 / 50 | 0 / 800 | 0 / 50 |
| `sea_ts_0p05__sea_bg_0p05__sea_cfg_0p10` | 0.05 | 0.05 | 0.10 | 537.810 | 0.972x | 37.457 | 34.81 | 0 / 50 | 0 / 752 | 6 / 44 |
| `sea_ts_0p05__sea_bg_0p05__sea_cfg_0p20` | 0.05 | 0.05 | 0.20 | 461.154 | 1.133x | 26.226 | 23.13 | 0 / 50 | 0 / 640 | 20 / 30 |
| `sea_ts_0p05__sea_bg_0p05__sea_cfg_0p40` | 0.05 | 0.05 | 0.40 | 394.925 | 1.323x | 20.084 | 18.75 | 0 / 50 | 0 / 544 | 32 / 18 |
| `sea_ts_0p05__sea_bg_0p05__sea_cfg_1p00` | 0.05 | 0.05 | 1.00 | 345.421 | 1.513x | 16.511 | 16.01 | 0 / 50 | 0 / 472 | 41 / 9 |
| `sea_ts_0p05__sea_bg_0p10__sea_cfg_0p05` | 0.05 | 0.10 | 0.05 | 529.685 | 0.987x | 37.465 | 35.93 | 0 / 50 | 68 / 732 | 0 / 50 |
| `sea_ts_0p05__sea_bg_0p10__sea_cfg_0p10` | 0.05 | 0.10 | 0.10 | 517.001 | 1.011x | 37.014 | 35.58 | 0 / 50 | 34 / 718 | 6 / 44 |
| `sea_ts_0p05__sea_bg_0p10__sea_cfg_0p20` | 0.05 | 0.10 | 0.20 | 440.022 | 1.188x | 26.390 | 23.13 | 0 / 50 | 34 / 606 | 20 / 30 |
| `sea_ts_0p05__sea_bg_0p10__sea_cfg_0p40` | 0.05 | 0.10 | 0.40 | 374.636 | 1.395x | 20.203 | 18.85 | 0 / 50 | 34 / 510 | 32 / 18 |
| `sea_ts_0p05__sea_bg_0p10__sea_cfg_1p00` | 0.05 | 0.10 | 1.00 | 324.602 | 1.610x | 16.544 | 16.03 | 0 / 50 | 34 / 438 | 41 / 9 |
| `sea_ts_0p05__sea_bg_0p20__sea_cfg_0p05` | 0.05 | 0.20 | 0.05 | 383.582 | 1.362x | 24.815 | 21.93 | 0 / 50 | 306 / 494 | 0 / 50 |
| `sea_ts_0p05__sea_bg_0p20__sea_cfg_0p10` | 0.05 | 0.20 | 0.10 | 370.953 | 1.409x | 24.615 | 21.89 | 0 / 50 | 273 / 479 | 6 / 44 |
| `sea_ts_0p05__sea_bg_0p20__sea_cfg_0p20` | 0.05 | 0.20 | 0.20 | 366.250 | 1.427x | 24.714 | 22.19 | 0 / 50 | 155 / 485 | 20 / 30 |
| `sea_ts_0p05__sea_bg_0p20__sea_cfg_0p40` | 0.05 | 0.20 | 0.40 | 301.202 | 1.735x | 19.788 | 18.73 | 0 / 50 | 153 / 391 | 32 / 18 |
| `sea_ts_0p05__sea_bg_0p20__sea_cfg_1p00` | 0.05 | 0.20 | 1.00 | 251.860 | 2.075x | 16.856 | 16.34 | 0 / 50 | 153 / 319 | 41 / 9 |
| `sea_ts_0p05__sea_bg_0p40__sea_cfg_0p05` | 0.05 | 0.40 | 0.05 | 257.488 | 2.030x | 19.417 | 18.12 | 0 / 50 | 512 / 288 | 0 / 50 |
| `sea_ts_0p05__sea_bg_0p40__sea_cfg_0p10` | 0.05 | 0.40 | 0.10 | 257.983 | 2.026x | 19.378 | 18.17 | 0 / 50 | 457 / 295 | 6 / 44 |
| `sea_ts_0p05__sea_bg_0p40__sea_cfg_0p20` | 0.05 | 0.40 | 0.20 | 244.832 | 2.135x | 19.454 | 18.08 | 0 / 50 | 353 / 287 | 20 / 30 |
| `sea_ts_0p05__sea_bg_0p40__sea_cfg_0p40` | 0.05 | 0.40 | 0.40 | 214.301 | 2.439x | 18.518 | 16.89 | 0 / 50 | 295 / 249 | 32 / 18 |
| `sea_ts_0p05__sea_bg_0p40__sea_cfg_1p00` | 0.05 | 0.40 | 1.00 | 188.678 | 2.770x | 18.973 | 18.08 | 0 / 50 | 256 / 216 | 41 / 9 |
| `sea_ts_0p05__sea_bg_1p00__sea_cfg_0p05` | 0.05 | 1.00 | 0.05 | 185.828 | 2.812x | 16.673 | 16.16 | 0 / 50 | 629 / 171 | 0 / 50 |
| `sea_ts_0p05__sea_bg_1p00__sea_cfg_0p10` | 0.05 | 1.00 | 0.10 | 177.897 | 2.938x | 16.549 | 16.07 | 0 / 50 | 588 / 164 | 6 / 44 |
| `sea_ts_0p05__sea_bg_1p00__sea_cfg_0p20` | 0.05 | 1.00 | 0.20 | 165.914 | 3.150x | 16.783 | 16.12 | 0 / 50 | 482 / 158 | 20 / 30 |
| `sea_ts_0p05__sea_bg_1p00__sea_cfg_0p40` | 0.05 | 1.00 | 0.40 | 153.697 | 3.400x | 16.954 | 16.36 | 0 / 50 | 394 / 150 | 32 / 18 |
| `sea_ts_0p05__sea_bg_1p00__sea_cfg_1p00` | 0.05 | 1.00 | 1.00 | 129.746 | 4.028x | 12.809 | 12.41 | 0 / 50 | 353 / 119 | 41 / 9 |
| `sea_ts_0p10__sea_bg_0p05__sea_cfg_0p05` | 0.10 | 0.05 | 0.05 | 511.664 | 1.021x | 36.303 | 34.41 | 6 / 44 | 0 / 704 | 0 / 50 |
| `sea_ts_0p10__sea_bg_0p05__sea_cfg_0p10` | 0.10 | 0.05 | 0.10 | 509.984 | 1.025x | 36.747 | 35.04 | 6 / 44 | 0 / 704 | 6 / 44 |
| `sea_ts_0p10__sea_bg_0p05__sea_cfg_0p20` | 0.10 | 0.05 | 0.20 | 432.758 | 1.208x | 26.430 | 23.14 | 6 / 46 | 0 / 592 | 20 / 30 |
| `sea_ts_0p10__sea_bg_0p05__sea_cfg_0p40` | 0.10 | 0.05 | 0.40 | 366.483 | 1.426x | 20.201 | 18.84 | 6 / 45 | 0 / 496 | 32 / 18 |
| `sea_ts_0p10__sea_bg_0p05__sea_cfg_1p00` | 0.10 | 0.05 | 1.00 | 316.594 | 1.651x | 16.548 | 16.02 | 6 / 44 | 0 / 424 | 41 / 9 |
| `sea_ts_0p10__sea_bg_0p10__sea_cfg_0p05` | 0.10 | 0.10 | 0.05 | 511.994 | 1.021x | 36.303 | 34.41 | 6 / 44 | 0 / 704 | 0 / 50 |
| `sea_ts_0p10__sea_bg_0p10__sea_cfg_0p10` | 0.10 | 0.10 | 0.10 | 509.720 | 1.025x | 36.747 | 35.04 | 6 / 44 | 0 / 704 | 6 / 44 |
| `sea_ts_0p10__sea_bg_0p10__sea_cfg_0p20` | 0.10 | 0.10 | 0.20 | 432.885 | 1.207x | 26.430 | 23.14 | 6 / 46 | 0 / 592 | 20 / 30 |
| `sea_ts_0p10__sea_bg_0p10__sea_cfg_0p40` | 0.10 | 0.10 | 0.40 | 366.605 | 1.426x | 20.201 | 18.84 | 6 / 45 | 0 / 496 | 32 / 18 |
| `sea_ts_0p10__sea_bg_0p10__sea_cfg_1p00` | 0.10 | 0.10 | 1.00 | 316.930 | 1.649x | 16.548 | 16.02 | 6 / 44 | 0 / 424 | 41 / 9 |
| `sea_ts_0p10__sea_bg_0p20__sea_cfg_0p05` | 0.10 | 0.20 | 0.05 | 365.327 | 1.431x | 24.774 | 22.02 | 6 / 44 | 240 / 464 | 0 / 50 |
| `sea_ts_0p10__sea_bg_0p20__sea_cfg_0p10` | 0.10 | 0.20 | 0.10 | 362.515 | 1.442x | 24.782 | 22.02 | 6 / 44 | 240 / 464 | 6 / 44 |
| `sea_ts_0p10__sea_bg_0p20__sea_cfg_0p20` | 0.10 | 0.20 | 0.20 | 357.675 | 1.461x | 24.768 | 22.24 | 6 / 46 | 122 / 470 | 20 / 30 |
| `sea_ts_0p10__sea_bg_0p20__sea_cfg_0p40` | 0.10 | 0.20 | 0.40 | 292.599 | 1.786x | 19.834 | 18.77 | 6 / 45 | 120 / 376 | 32 / 18 |
| `sea_ts_0p10__sea_bg_0p20__sea_cfg_1p00` | 0.10 | 0.20 | 1.00 | 242.957 | 2.151x | 16.928 | 16.39 | 6 / 44 | 120 / 304 | 41 / 9 |
| `sea_ts_0p10__sea_bg_0p40__sea_cfg_0p05` | 0.10 | 0.40 | 0.05 | 265.169 | 1.971x | 19.322 | 18.07 | 6 / 44 | 402 / 302 | 0 / 50 |
| `sea_ts_0p10__sea_bg_0p40__sea_cfg_0p10` | 0.10 | 0.40 | 0.10 | 262.801 | 1.989x | 19.323 | 18.07 | 6 / 44 | 402 / 302 | 6 / 44 |
| `sea_ts_0p10__sea_bg_0p40__sea_cfg_0p20` | 0.10 | 0.40 | 0.20 | 249.273 | 2.097x | 19.519 | 18.18 | 6 / 46 | 298 / 294 | 20 / 30 |
| `sea_ts_0p10__sea_bg_0p40__sea_cfg_0p40` | 0.10 | 0.40 | 0.40 | 223.408 | 2.339x | 19.296 | 17.63 | 6 / 45 | 232 / 264 | 32 / 18 |
| `sea_ts_0p10__sea_bg_0p40__sea_cfg_1p00` | 0.10 | 0.40 | 1.00 | 192.856 | 2.710x | 18.897 | 18.05 | 6 / 44 | 201 / 223 | 41 / 9 |
| `sea_ts_0p10__sea_bg_1p00__sea_cfg_0p05` | 0.10 | 1.00 | 0.05 | 176.004 | 2.969x | 16.560 | 16.00 | 6 / 44 | 547 / 157 | 0 / 50 |
| `sea_ts_0p10__sea_bg_1p00__sea_cfg_0p10` | 0.10 | 1.00 | 0.10 | 173.612 | 3.010x | 16.559 | 16.00 | 6 / 44 | 547 / 157 | 6 / 44 |
| `sea_ts_0p10__sea_bg_1p00__sea_cfg_0p20` | 0.10 | 1.00 | 0.20 | 161.376 | 3.238x | 16.795 | 16.08 | 6 / 46 | 441 / 151 | 20 / 30 |
| `sea_ts_0p10__sea_bg_1p00__sea_cfg_0p40` | 0.10 | 1.00 | 0.40 | 149.233 | 3.502x | 16.846 | 16.14 | 6 / 45 | 353 / 143 | 32 / 18 |
| `sea_ts_0p10__sea_bg_1p00__sea_cfg_1p00` | 0.10 | 1.00 | 1.00 | 129.182 | 4.045x | 12.810 | 12.42 | 5 / 45 | 313 / 119 | 41 / 9 |
| `sea_ts_0p20__sea_bg_0p05__sea_cfg_0p05` | 0.20 | 0.05 | 0.05 | 369.481 | 1.414x | 24.558 | 21.98 | 20 / 30 | 0 / 480 | 0 / 50 |
| `sea_ts_0p20__sea_bg_0p05__sea_cfg_0p10` | 0.20 | 0.05 | 0.10 | 361.993 | 1.444x | 24.433 | 21.84 | 21 / 32 | 0 / 472 | 6 / 44 |
| `sea_ts_0p20__sea_bg_0p05__sea_cfg_0p20` | 0.20 | 0.05 | 0.20 | 351.073 | 1.489x | 24.848 | 22.11 | 22 / 30 | 0 / 464 | 20 / 30 |
| `sea_ts_0p20__sea_bg_0p05__sea_cfg_0p40` | 0.20 | 0.05 | 0.40 | 294.535 | 1.774x | 19.773 | 18.75 | 20 / 37 | 0 / 384 | 32 / 18 |
| `sea_ts_0p20__sea_bg_0p05__sea_cfg_1p00` | 0.20 | 0.05 | 1.00 | 245.261 | 2.131x | 16.848 | 16.32 | 20 / 32 | 0 / 312 | 41 / 9 |
| `sea_ts_0p20__sea_bg_0p10__sea_cfg_0p05` | 0.20 | 0.10 | 0.05 | 369.657 | 1.414x | 24.558 | 21.98 | 20 / 30 | 0 / 480 | 0 / 50 |
| `sea_ts_0p20__sea_bg_0p10__sea_cfg_0p10` | 0.20 | 0.10 | 0.10 | 362.315 | 1.442x | 24.433 | 21.84 | 21 / 32 | 0 / 472 | 6 / 44 |
| `sea_ts_0p20__sea_bg_0p10__sea_cfg_0p20` | 0.20 | 0.10 | 0.20 | 351.063 | 1.489x | 24.848 | 22.11 | 22 / 30 | 0 / 464 | 20 / 30 |
| `sea_ts_0p20__sea_bg_0p10__sea_cfg_0p40` | 0.20 | 0.10 | 0.40 | 294.773 | 1.773x | 19.773 | 18.75 | 20 / 37 | 0 / 384 | 32 / 18 |
| `sea_ts_0p20__sea_bg_0p10__sea_cfg_1p00` | 0.20 | 0.10 | 1.00 | 245.180 | 2.132x | 16.848 | 16.32 | 20 / 32 | 0 / 312 | 41 / 9 |
| `sea_ts_0p20__sea_bg_0p20__sea_cfg_0p05` | 0.20 | 0.20 | 0.05 | 367.173 | 1.423x | 24.599 | 22.17 | 20 / 30 | 4 / 476 | 0 / 50 |
| `sea_ts_0p20__sea_bg_0p20__sea_cfg_0p10` | 0.20 | 0.20 | 0.10 | 359.623 | 1.453x | 24.397 | 22.00 | 21 / 32 | 4 / 468 | 6 / 44 |
| `sea_ts_0p20__sea_bg_0p20__sea_cfg_0p20` | 0.20 | 0.20 | 0.20 | 349.338 | 1.496x | 24.898 | 22.12 | 22 / 30 | 2 / 462 | 20 / 30 |
| `sea_ts_0p20__sea_bg_0p20__sea_cfg_0p40` | 0.20 | 0.20 | 0.40 | 293.726 | 1.779x | 19.847 | 18.78 | 20 / 37 | 2 / 382 | 32 / 18 |
| `sea_ts_0p20__sea_bg_0p20__sea_cfg_1p00` | 0.20 | 0.20 | 1.00 | 243.932 | 2.142x | 16.868 | 16.34 | 20 / 32 | 2 / 310 | 41 / 9 |
| `sea_ts_0p20__sea_bg_0p40__sea_cfg_0p05` | 0.20 | 0.40 | 0.05 | 250.148 | 2.089x | 19.407 | 18.06 | 20 / 30 | 194 / 286 | 0 / 50 |
| `sea_ts_0p20__sea_bg_0p40__sea_cfg_0p10` | 0.20 | 0.40 | 0.10 | 242.447 | 2.156x | 19.294 | 18.10 | 21 / 32 | 194 / 278 | 6 / 44 |
| `sea_ts_0p20__sea_bg_0p40__sea_cfg_0p20` | 0.20 | 0.40 | 0.20 | 237.225 | 2.203x | 19.514 | 18.26 | 22 / 30 | 185 / 279 | 20 / 30 |
| `sea_ts_0p20__sea_bg_0p40__sea_cfg_0p40` | 0.20 | 0.40 | 0.40 | 219.796 | 2.378x | 18.707 | 17.46 | 20 / 37 | 122 / 262 | 32 / 18 |
| `sea_ts_0p20__sea_bg_0p40__sea_cfg_1p00` | 0.20 | 0.40 | 1.00 | 183.049 | 2.855x | 18.412 | 17.47 | 20 / 32 | 97 / 215 | 41 / 9 |
| `sea_ts_0p20__sea_bg_1p00__sea_cfg_0p05` | 0.20 | 1.00 | 0.05 | 161.009 | 3.246x | 16.856 | 16.32 | 20 / 30 | 336 / 144 | 0 / 50 |
| `sea_ts_0p20__sea_bg_1p00__sea_cfg_0p10` | 0.20 | 1.00 | 0.10 | 159.535 | 3.276x | 16.858 | 16.32 | 21 / 32 | 328 / 144 | 6 / 44 |
| `sea_ts_0p20__sea_bg_1p00__sea_cfg_0p20` | 0.20 | 1.00 | 0.20 | 152.502 | 3.427x | 16.863 | 16.33 | 22 / 30 | 320 / 144 | 20 / 30 |
| `sea_ts_0p20__sea_bg_1p00__sea_cfg_0p40` | 0.20 | 1.00 | 0.40 | 140.418 | 3.722x | 16.973 | 16.50 | 20 / 37 | 248 / 136 | 32 / 18 |
| `sea_ts_0p20__sea_bg_1p00__sea_cfg_1p00` | 0.20 | 1.00 | 1.00 | 125.250 | 4.172x | 13.207 | 12.73 | 20 / 31 | 193 / 119 | 41 / 9 |
| `sea_ts_0p40__sea_bg_0p05__sea_cfg_0p05` | 0.40 | 0.05 | 0.05 | 246.592 | 2.119x | 19.559 | 18.20 | 32 / 18 | 0 / 288 | 0 / 50 |
| `sea_ts_0p40__sea_bg_0p05__sea_cfg_0p10` | 0.40 | 0.05 | 0.10 | 244.210 | 2.140x | 19.609 | 18.33 | 34 / 21 | 0 / 288 | 6 / 44 |
| `sea_ts_0p40__sea_bg_0p05__sea_cfg_0p20` | 0.40 | 0.05 | 0.20 | 233.377 | 2.239x | 19.352 | 17.99 | 33 / 25 | 0 / 280 | 20 / 30 |
| `sea_ts_0p40__sea_bg_0p05__sea_cfg_0p40` | 0.40 | 0.05 | 0.40 | 207.875 | 2.514x | 18.512 | 17.09 | 37 / 18 | 0 / 248 | 32 / 18 |
| `sea_ts_0p40__sea_bg_0p05__sea_cfg_1p00` | 0.40 | 0.05 | 1.00 | 184.227 | 2.837x | 19.007 | 18.05 | 32 / 22 | 0 / 216 | 41 / 9 |
| `sea_ts_0p40__sea_bg_0p10__sea_cfg_0p05` | 0.40 | 0.10 | 0.05 | 247.355 | 2.113x | 19.559 | 18.20 | 32 / 18 | 0 / 288 | 0 / 50 |
| `sea_ts_0p40__sea_bg_0p10__sea_cfg_0p10` | 0.40 | 0.10 | 0.10 | 244.961 | 2.133x | 19.609 | 18.33 | 34 / 21 | 0 / 288 | 6 / 44 |
| `sea_ts_0p40__sea_bg_0p10__sea_cfg_0p20` | 0.40 | 0.10 | 0.20 | 233.824 | 2.235x | 19.352 | 17.99 | 33 / 25 | 0 / 280 | 20 / 30 |
| `sea_ts_0p40__sea_bg_0p10__sea_cfg_0p40` | 0.40 | 0.10 | 0.40 | 207.994 | 2.513x | 18.512 | 17.09 | 37 / 18 | 0 / 248 | 32 / 18 |
| `sea_ts_0p40__sea_bg_0p10__sea_cfg_1p00` | 0.40 | 0.10 | 1.00 | 183.661 | 2.845x | 19.007 | 18.05 | 32 / 22 | 0 / 216 | 41 / 9 |
| `sea_ts_0p40__sea_bg_0p20__sea_cfg_0p05` | 0.40 | 0.20 | 0.05 | 246.919 | 2.116x | 19.559 | 18.20 | 32 / 18 | 0 / 288 | 0 / 50 |
| `sea_ts_0p40__sea_bg_0p20__sea_cfg_0p10` | 0.40 | 0.20 | 0.10 | 244.585 | 2.137x | 19.609 | 18.33 | 34 / 21 | 0 / 288 | 6 / 44 |
| `sea_ts_0p40__sea_bg_0p20__sea_cfg_0p20` | 0.40 | 0.20 | 0.20 | 233.340 | 2.240x | 19.352 | 17.99 | 33 / 25 | 0 / 280 | 20 / 30 |
| `sea_ts_0p40__sea_bg_0p20__sea_cfg_0p40` | 0.40 | 0.20 | 0.40 | 208.057 | 2.512x | 18.512 | 17.09 | 37 / 18 | 0 / 248 | 32 / 18 |
| `sea_ts_0p40__sea_bg_0p20__sea_cfg_1p00` | 0.40 | 0.20 | 1.00 | 183.683 | 2.845x | 19.007 | 18.05 | 32 / 22 | 0 / 216 | 41 / 9 |
| `sea_ts_0p40__sea_bg_0p40__sea_cfg_0p05` | 0.40 | 0.40 | 0.05 | 212.476 | 2.460x | 18.749 | 17.48 | 32 / 18 | 56 / 232 | 0 / 50 |
| `sea_ts_0p40__sea_bg_0p40__sea_cfg_0p10` | 0.40 | 0.40 | 0.10 | 214.260 | 2.439x | 18.851 | 17.60 | 34 / 21 | 49 / 239 | 6 / 44 |
| `sea_ts_0p40__sea_bg_0p40__sea_cfg_0p20` | 0.40 | 0.40 | 0.20 | 212.399 | 2.460x | 19.046 | 17.44 | 33 / 25 | 34 / 246 | 20 / 30 |
| `sea_ts_0p40__sea_bg_0p40__sea_cfg_0p40` | 0.40 | 0.40 | 0.40 | 197.633 | 2.644x | 18.736 | 17.41 | 36 / 18 | 25 / 231 | 32 / 18 |
| `sea_ts_0p40__sea_bg_0p40__sea_cfg_1p00` | 0.40 | 0.40 | 1.00 | 172.320 | 3.033x | 18.152 | 17.35 | 32 / 22 | 19 / 197 | 41 / 9 |
| `sea_ts_0p40__sea_bg_1p00__sea_cfg_0p05` | 0.40 | 1.00 | 0.05 | 148.328 | 3.523x | 16.666 | 16.11 | 32 / 18 | 160 / 128 | 0 / 50 |
| `sea_ts_0p40__sea_bg_1p00__sea_cfg_0p10` | 0.40 | 1.00 | 0.10 | 145.798 | 3.584x | 16.676 | 16.08 | 34 / 21 | 160 / 128 | 6 / 44 |
| `sea_ts_0p40__sea_bg_1p00__sea_cfg_0p20` | 0.40 | 1.00 | 0.20 | 139.702 | 3.741x | 16.606 | 15.99 | 33 / 25 | 152 / 128 | 20 / 30 |
| `sea_ts_0p40__sea_bg_1p00__sea_cfg_0p40` | 0.40 | 1.00 | 0.40 | 134.159 | 3.895x | 16.662 | 16.10 | 36 / 18 | 128 / 128 | 32 / 18 |
| `sea_ts_0p40__sea_bg_1p00__sea_cfg_1p00` | 0.40 | 1.00 | 1.00 | 114.801 | 4.552x | 14.469 | 13.84 | 32 / 22 | 112 / 104 | 41 / 9 |
| `sea_ts_1p00__sea_bg_0p05__sea_cfg_0p05` | 1.00 | 0.05 | 0.05 | 154.654 | 3.379x | 18.527 | 17.11 | 41 / 9 | 0 / 144 | 0 / 50 |
| `sea_ts_1p00__sea_bg_0p05__sea_cfg_0p10` | 1.00 | 0.05 | 0.10 | 152.242 | 3.433x | 18.527 | 17.11 | 41 / 9 | 0 / 144 | 6 / 44 |
| `sea_ts_1p00__sea_bg_0p05__sea_cfg_0p20` | 1.00 | 0.05 | 0.20 | 146.186 | 3.575x | 18.233 | 17.35 | 42 / 11 | 0 / 144 | 20 / 30 |
| `sea_ts_1p00__sea_bg_0p05__sea_cfg_0p40` | 1.00 | 0.05 | 0.40 | 136.271 | 3.835x | 17.728 | 16.83 | 42 / 12 | 0 / 136 | 32 / 18 |
| `sea_ts_1p00__sea_bg_0p05__sea_cfg_1p00` | 1.00 | 0.05 | 1.00 | 117.268 | 4.456x | 13.918 | 13.54 | 42 / 11 | 0 / 112 | 41 / 9 |
| `sea_ts_1p00__sea_bg_0p10__sea_cfg_0p05` | 1.00 | 0.10 | 0.05 | 154.634 | 3.380x | 18.527 | 17.11 | 41 / 9 | 0 / 144 | 0 / 50 |
| `sea_ts_1p00__sea_bg_0p10__sea_cfg_0p10` | 1.00 | 0.10 | 0.10 | 152.208 | 3.433x | 18.527 | 17.11 | 41 / 9 | 0 / 144 | 6 / 44 |
| `sea_ts_1p00__sea_bg_0p10__sea_cfg_0p20` | 1.00 | 0.10 | 0.20 | 146.410 | 3.569x | 18.233 | 17.35 | 42 / 11 | 0 / 144 | 20 / 30 |
| `sea_ts_1p00__sea_bg_0p10__sea_cfg_0p40` | 1.00 | 0.10 | 0.40 | 136.195 | 3.837x | 17.728 | 16.83 | 42 / 12 | 0 / 136 | 32 / 18 |
| `sea_ts_1p00__sea_bg_0p10__sea_cfg_1p00` | 1.00 | 0.10 | 1.00 | 117.361 | 4.453x | 13.918 | 13.54 | 42 / 11 | 0 / 112 | 41 / 9 |
| `sea_ts_1p00__sea_bg_0p20__sea_cfg_0p05` | 1.00 | 0.20 | 0.05 | 155.365 | 3.364x | 18.527 | 17.11 | 41 / 9 | 0 / 144 | 0 / 50 |
| `sea_ts_1p00__sea_bg_0p20__sea_cfg_0p10` | 1.00 | 0.20 | 0.10 | 152.859 | 3.419x | 18.527 | 17.11 | 41 / 9 | 0 / 144 | 6 / 44 |
| `sea_ts_1p00__sea_bg_0p20__sea_cfg_0p20` | 1.00 | 0.20 | 0.20 | 147.076 | 3.553x | 18.233 | 17.35 | 42 / 11 | 0 / 144 | 20 / 30 |
| `sea_ts_1p00__sea_bg_0p20__sea_cfg_0p40` | 1.00 | 0.20 | 0.40 | 137.101 | 3.812x | 17.728 | 16.83 | 42 / 12 | 0 / 136 | 32 / 18 |
| `sea_ts_1p00__sea_bg_0p20__sea_cfg_1p00` | 1.00 | 0.20 | 1.00 | 117.845 | 4.435x | 13.918 | 13.54 | 42 / 11 | 0 / 112 | 41 / 9 |
| `sea_ts_1p00__sea_bg_0p40__sea_cfg_0p05` | 1.00 | 0.40 | 0.05 | 154.972 | 3.372x | 18.527 | 17.11 | 41 / 9 | 0 / 144 | 0 / 50 |
| `sea_ts_1p00__sea_bg_0p40__sea_cfg_0p10` | 1.00 | 0.40 | 0.10 | 152.478 | 3.427x | 18.527 | 17.11 | 41 / 9 | 0 / 144 | 6 / 44 |
| `sea_ts_1p00__sea_bg_0p40__sea_cfg_0p20` | 1.00 | 0.40 | 0.20 | 146.456 | 3.568x | 18.233 | 17.35 | 42 / 11 | 0 / 144 | 20 / 30 |
| `sea_ts_1p00__sea_bg_0p40__sea_cfg_0p40` | 1.00 | 0.40 | 0.40 | 136.306 | 3.834x | 17.728 | 16.83 | 42 / 12 | 0 / 136 | 32 / 18 |
| `sea_ts_1p00__sea_bg_0p40__sea_cfg_1p00` | 1.00 | 0.40 | 1.00 | 117.366 | 4.453x | 13.918 | 13.54 | 42 / 11 | 0 / 112 | 41 / 9 |
| `sea_ts_1p00__sea_bg_1p00__sea_cfg_0p05` | 1.00 | 1.00 | 0.05 | 115.437 | 4.527x | 15.608 | 15.20 | 41 / 9 | 64 / 80 | 0 / 50 |
| `sea_ts_1p00__sea_bg_1p00__sea_cfg_0p10` | 1.00 | 1.00 | 0.10 | 112.943 | 4.627x | 15.608 | 15.21 | 41 / 9 | 64 / 80 | 6 / 44 |
| `sea_ts_1p00__sea_bg_1p00__sea_cfg_0p20` | 1.00 | 1.00 | 0.20 | 107.238 | 4.873x | 15.633 | 15.24 | 42 / 11 | 64 / 80 | 20 / 30 |
| `sea_ts_1p00__sea_bg_1p00__sea_cfg_0p40` | 1.00 | 1.00 | 0.40 | 101.941 | 5.127x | 13.960 | 13.78 | 42 / 12 | 56 / 80 | 32 / 18 |
| `sea_ts_1p00__sea_bg_1p00__sea_cfg_1p00` | 1.00 | 1.00 | 1.00 | 92.597 | 5.644x | 11.914 | 11.66 | 42 / 11 | 40 / 72 | 41 / 9 |
