# Wan2.2 Supplementary Cache Experiments Report

生成时间：2026-06-10 CST

本报告整理主报告之外的补充、诊断和开发实验。主报告只包含 fixed ZEUS 10 prompts、ZEUS-threshold reuse_interp 10 prompts、three-cache 64 threshold grid。

## 1. Shared Setting

除 smoke test 特别说明外，补充实验通常使用 Wan2.2 T2V-A14B、prompt 01、seed `42`、`832*480`、`45` frames、`50` dpm++ steps、no-cache baseline、FFmpeg PSNR、ffprobe 验证视频 metadata。

## 2. Smoke ZEUS Validation

Result root: `experiment_results/wan22_smoke_zeus_20260608_0951`

Setting: prompt 01, seed `20260608`, `5` frames, `8` steps, ZEUS acc range `3-8`, denominator `2`, modular `(1,)`, `reuse_interp`; this is only a pipeline validation run.

| prompt | baseline s | ZEUS s | speedup | mean PSNR dB | reuse/recompute |
|---|---|---|---|---|---|
| 01 | 375.000 | 374.000 | 1.003 | 14.917 | 4/12 |

Conclusion: 只用于验证环境、生成路径、ffprobe 和 PSNR 流程，不用于正式质量/速度判断。

## 3. ZEUS-threshold Prompt 01, 7-threshold Pilot

Result root: `experiment_results/wan22_zeus_threshold_prompt01_7th_20260608_162827`

Setting: prompt 01, seed `42`, `zeus-threshold`, thresholds `0.001`, `0.005`, `0.02`, `0.08`, `0.20`, `0.60`, `1.50`, `reuse_interp`.

| threshold | baseline s | candidate s | speedup | mean PSNR dB | min PSNR dB | reuse/recompute |
|---|---|---|---|---|---|---|
| 0.001 | 522.603 | 522.711 | 1.000 | Infinity | Infinity | 0/50 |
| 0.005 | 522.603 | 470.287 | 1.111 | 26.954 | 22.200 | 5/45 |
| 0.02 | 522.603 | 325.253 | 1.607 | 18.606 | 17.350 | 19/31 |
| 0.08 | 522.603 | 232.145 | 2.251 | 18.873 | 17.850 | 28/22 |
| 0.20 | 522.603 | 201.093 | 2.599 | 18.900 | 17.850 | 31/19 |
| 0.60 | 522.603 | 191.124 | 2.734 | 18.928 | 17.900 | 32/18 |
| 1.50 | 522.603 | 191.336 | 2.731 | 18.928 | 17.900 | 32/18 |

Conclusion: `0.005` 是 prompt 01 上唯一高质量 threshold 点；`0.02+` 进入明显降质区间；复用在高阈值处由于 guard/schedule 结构饱和。

## 4. ZEUS-threshold Prompt 01, Timestep-aware Interpolation

Result root: `experiment_results/wan22_zeus_threshold_taware_prompt01_5th_20260608_191714`

Setting: prompt 01, seed `42`, thresholds `0.005`, `0.02`, `0.08`, `0.20`, `0.60`; interpolation changed to timestep-aware scale between recent recompute step indices.

| threshold | baseline s | candidate s | speedup | mean PSNR dB | min PSNR dB | reuse/recompute |
|---|---|---|---|---|---|---|
| 0.005 | 522.603 | 470.918 | 1.110 | 28.196 | 24.160 | 5/45 |
| 0.02 | 522.603 | 323.022 | 1.618 | 18.075 | 17.360 | 19/31 |
| 0.08 | 522.603 | 230.943 | 2.263 | 17.782 | 17.140 | 28/22 |
| 0.20 | 522.603 | 199.447 | 2.620 | 17.666 | 17.010 | 31/19 |
| 0.60 | 522.603 | 189.071 | 2.764 | 17.622 | 16.980 | 32/18 |

Conclusion: timestep-aware interpolation improves conservative `0.005` quality, but larger thresholds worse than reuse_interp；当前主线继续使用 reuse_interp。

## 5. Block Cache Only

Result root: `experiment_results/wan22_block_cache_only_50step_45f_480p_20260609_125436`

Setting: prompt 01, seed `42`; timestep cache disabled, CFG cache disabled. Compared BWCache thresholds `0.05`, `0.15`, `0.30` and block-group thresholds `0.01`, `0.03`, `0.05`; block-group uses group size `5`, pooled rel-L1, start/end `0.1/0.9`, max reuse `3`.

| method | threshold | baseline s | candidate s | speedup | mean PSNR dB | min PSNR dB | reuse/recompute |
|---|---|---|---|---|---|---|---|
| block_group | 0.01 | 522.603 | 532.389 | 0.982 | Infinity | Infinity | 0/800 |
| block_group | 0.03 | 522.603 | 384.116 | 1.361 | 19.396 | 17.800 | 244/556 |
| block_group | 0.05 | 522.603 | 305.079 | 1.713 | 19.491 | 18.040 | 374/426 |
| bwcache | 0.05 | 522.603 | 507.238 | 1.030 | 28.895 | 25.140 | 4/96 |
| bwcache | 0.15 | 522.603 | 327.192 | 1.597 | 18.370 | 17.100 | 41/59 |
| bwcache | 0.30 | 522.603 | 312.253 | 1.674 | 17.632 | 16.980 | 44/56 |

Conclusion: BWCache `0.05` 质量高但几乎不加速；block-group `0.03~0.05` 更适合作为速度/质量 tradeoff 的组合 cache 方向。

## 6. Three-cache Development Runs

These runs used prompt 01 with timestep `zeus-threshold=0.02`, block-group `0.03`, CFG threshold `0.03`, while CFG policy evolved.

| run | baseline s | candidate s | speedup | mean PSNR dB | timestep branch reuse/recompute | block group reuse/recompute | CFG reuse/recompute | setting note |
|---|---|---|---|---|---|---|---|---|
| initial merge | 522.603 | 357.403 | 1.462 | 18.001 | 19/70 | 26/534 | 11/39 | old CFG behavior, force uncond recompute on miss |
| internal metric + full refresh | 522.603 | 492.713 | 1.061 | 19.623 | 24/92 | 26/710 | 17/33 | CFG metric `timestep_modulated_input_rel_l1`, full cond/uncond refresh on miss |
| optimized scheduling | 522.603 | 381.459 | 1.370 | 19.603 | 12/71 | 26/542 | 17/33 | decide CFG hit/miss before cond for input-feature metric |

Conclusion: internal CFG metric + full refresh improved quality but was too slow; scheduling optimization recovered speed with almost no PSNR loss. Later threshold-grid experiments disabled CFG miss forced recompute.

## 7. Cache Ablation

Result root: `experiment_results/wan22_cache_ablation_prompt01_50step_45f_480p_20260609_184625`

Setting: prompt 01, seed `42`, single-process runner; timestep `zeus-threshold=0.02`, block-group `0.03`, CFG threshold `0.03`, CFG internal metric. This ablation was run under the full-refresh CFG policy present at that time.

| candidate | timestep | block-group | CFG | baseline s | candidate s | speedup | mean PSNR dB | min PSNR dB |
|---|---|---|---|---|---|---|---|---|
| timestep_only | True | False | False | 522.603 | 326.641 | 1.600 | 18.606 | 17.350 |
| block_only | False | True | False | 522.603 | 383.713 | 1.362 | 19.396 | 17.800 |
| cfg_only | False | False | True | 522.603 | 455.366 | 1.148 | 21.571 | 20.310 |
| timestep_block | True | True | False | 522.603 | 298.889 | 1.748 | 18.159 | 17.210 |
| timestep_cfg | True | False | True | 522.603 | 392.466 | 1.332 | 20.910 | 19.670 |
| block_cfg | False | True | True | 522.603 | 386.542 | 1.352 | 19.446 | 17.970 |
| all_three | True | True | True | 522.603 | 381.343 | 1.370 | 19.603 | 18.100 |

Conclusion: `timestep_only` here is threshold `0.02`, not fixed ZEUS. CFG alone is relatively mild；timestep threshold `0.02` 是主要低 PSNR 来源。CFG 能恢复一部分 timestep-only 质量，但会牺牲速度。

## 8. Failed or Superseded Runs

| Experiment | Reason |
|---|---|
| `wan22_zeus_timestep_cache_50step_45f_480p_full_20260608_104151` | early failed baseline attempt; superseded |
| `wan22_zeus_timestep_cache_50step_45f_480p_full_20260608_104311` | interim prompt 01 run; superseded by completed 114307 run |
| block-cache retry histories under `failed_history/` | diagnosed OOM and BWCache tail-len bug before corrected block-cache-only summary |

## 9. Overall Supplementary Takeaways

- Prompt 01 threshold pilots explain why `0.005` is the conservative timestep threshold and why `0.02+` is risky.
- Timestep-aware interpolation is not the current default because only the conservative point improved.
- Block-group is the preferred block-cache direction for combined experiments, but aggressive block thresholds alone sit near `19 dB`.
- CFG full-refresh-on-miss is too expensive for the current main grid direction; current planned experiments keep it disabled.
