# Wan2.2 T2V-14B Cache Acceleration Report

生成时间：2026-06-10 CST

本文汇总 `/hy-tmp/work/Wan2.2/experiment_results` 下已有实验结果，重点记录有完整配置、视频、ffprobe、PSNR 和 summary/aggregate 表的实验。除特殊说明外，正式比较实验均使用 Wan2.2 T2V-A14B、prompt 01 或 `prompt.txt` 的 10 个 prompts、seed `42`、`832*480`、`45` frames、`50` dpm++ steps、`offload_model=True`、`convert_model_dtype`，并以同 prompt/seed/shape 的 no-cache baseline 作为 PSNR 参考。

## 1. Executive Summary

- 固定 cadence 的 ZEUS timestep cache 在 10 prompts 上给出稳定的高速度：总体 `1.986x`，平均 PSNR `23.705 dB`。这是目前多 prompt 上最完整、最可信的基准加速结果。
- `zeus-threshold` timestep cache 的 prompt 01 threshold sweep 显示 `0.005` 是高质量点，`1.111x / 26.954 dB`；`0.02` 开始明显降质，prompt 01 为 `1.607x / 18.606 dB`。
- 10 prompts 的 `zeus-threshold reuse_interp` 结果表明阈值增大后速度可以到 `2.75x`，但平均 PSNR 约 `21 dB`，且最差 prompt 可低到 `13-14 dB`，稳定性不足。
- block cache 单独测试中，`block-group` 比 BWCache 更适合当前方向：`block-group 0.05` 达到 `1.713x / 19.491 dB`，优于激进 BWCache 的速度质量折中；但高质量 block cache 点仍然只有 BWCache `0.05`，`1.030x / 28.895 dB`。
- CFG cache 单独启用在 prompt 01 上是相对温和的加速点：`1.148x / 21.571 dB`。在三缓存组合中，CFG 可改善 timestep-only 的质量，但如果开启 miss 强制全量重算会显著损失速度。目前已关闭 CFG miss forced recompute。
- 最新三缓存 64 组合 threshold grid 已全部完成。最快组合 `ts=0.6, block=1.0, cfg=1.0` 达到 `4.080x`，但 PSNR 仅 `15.225 dB`。更有价值的候选是：
  - PSNR `>=25 dB`：`ts=0.005, block=0.001, cfg=0.001`，`1.039x / 26.954 dB`
  - PSNR `>=22 dB`：`ts=0.005, block=0.015, cfg=0.03`，`1.204x / 23.448 dB`
  - PSNR `>=20 dB`：`ts=0.005, block=0.03, cfg=0.02`，`1.369x / 20.042 dB`

当前建议：若追求较稳质量，下一轮应围绕 `timestep=0.005`、`block_group=0.015~0.03`、`cfg=0.02~0.03` 做更细粒度搜索，并扩展到 10 prompts；若追求速度上限，可继续研究 `block_group=1.0` 与高 timestep threshold 的低质量高速度区域，但其质量目前不适合作为推荐方案。

## 2. Shared Evaluation Protocol

### 2.1 Common Generation Settings

| Item | Value |
|---|---|
| Project root | `/hy-tmp/work/Wan2.2` |
| Model | Wan2.2 T2V-A14B |
| Checkpoint | `/hy-tmp/models/Wan2.2-T2V-A14B` |
| Task | `t2v-A14B` |
| Resolution | `832*480` |
| Frame count | `45` for formal comparisons; smoke test used `5` |
| FPS | `16` |
| Duration | `2.812500s` for 45-frame runs |
| Sampling steps | `50` for formal comparisons |
| Solver | `dpm++` |
| Seed | `42` for aligned experiments |
| Baseline | no-cache, same prompt/seed/shape |
| Timing | `inference_compute_elapsed_seconds` |
| PSNR | FFmpeg `psnr` filter, perfect frames over `100 dB` excluded |
| Archive root | `/hy-tmp/wan22_*`, symlinked under `experiment_results/` |

### 2.2 Implemented Cache Types

| Cache | Implementation | Main knobs |
|---|---|---|
| Timestep cache | `zeus` fixed cadence | `acc_range`, denominator/modular, `reuse_interp`, `max_interval`, lagrange settings |
| Timestep threshold cache | `zeus-threshold` | latent relative-L1 threshold, `acc_range=(8,47)`, `reuse_interp`, `max_interval=6` |
| BWCache block cache | `bwcache` | threshold, reuse interval, tail/start/end guards, pooled feature metric |
| Block-group cache | `block-group` | group size `5`, pooled relative-L1 threshold, start/end `0.1/0.9`, max reuse `3` |
| CFG cache | `threshold` | metric `timestep_modulated_input_rel_l1`, threshold, start/end `0.1/0.9`, max reuse `3` |

Important implementation state:

- CFG cache is the outer branch-selection logic.
- If CFG hits, it skips `uncond` and reuses cached CFG delta.
- If CFG misses, current planned experiments do not force bypassing timestep/block caches. `force_uncond_recompute_on_miss=False`.
- For branches that actually run, timestep cache is checked before block cache.
- Explicit branch state is used: `cond` / `uncond`; no call-parity inference.

## 3. Experiment Inventory

Scanned symlinks under `/hy-tmp/work/Wan2.2/experiment_results`:

| Experiment | Status | Notes |
|---|---|---|
| `wan22_smoke_zeus_20260608_0951` | completed smoke | 5 frames, 8 steps only; pipeline validation |
| `wan22_zeus_timestep_cache_50step_45f_480p_full_20260608_104151` | failed/incomplete | early failed baseline attempt |
| `wan22_zeus_timestep_cache_50step_45f_480p_full_20260608_104311` | stopped/interim | prompt 01 interim, superseded by 114307 |
| `wan22_zeus_timestep_cache_50step_45f_480p_full_20260608_114307` | completed | formal fixed ZEUS, 10 prompts |
| `wan22_zeus_threshold_prompt01_7th_20260608_162827` | completed | prompt 01, 7 threshold pilot |
| `wan22_zeus_threshold_taware_prompt01_5th_20260608_191714` | completed | prompt 01, timestep-aware interpolation |
| `wan22_zeus_threshold_reuse_interp_10prompt_5th_20260608_195427` | completed | 10 prompts, 5 threshold reuse_interp |
| `wan22_block_cache_only_50step_45f_480p_20260609_125436` | completed | prompt 01, BWCache vs block-group |
| `wan22_merge_three_cache_prompt01_20260609_153438` | completed | first three-cache smoke, old CFG behavior |
| `wan22_merge_three_cache_internal_metric_prompt01_20260609_162556` | completed | internal CFG metric + full refresh |
| `wan22_merge_three_cache_optimized_schedule_prompt01_20260609_164735` | completed | internal CFG metric + optimized scheduling |
| `wan22_cache_ablation_prompt01_50step_45f_480p_20260609_184625` | completed | prompt 01, 7 cache ablations |
| `wan22_three_cache_threshold_grid_prompt01_50step_45f_480p_20260610_012518` | completed | prompt 01, 64 threshold combinations |

## 4. Fixed ZEUS Timestep Cache

### 4.1 Smoke Test

Result root: `experiment_results/wan22_smoke_zeus_20260608_0951`

Purpose: validate environment, generation path, ffprobe, timing, and PSNR archive.

Config:

| Item | Value |
|---|---|
| prompt | prompt 01 |
| seed | `20260608` |
| frames | `5` |
| steps | `8` |
| ZEUS settings | `acc_start=3`, `acc_end=8`, denominator `2`, modular `(1,)`, `reuse_interp`, lagrange term `0` |

Result:

| baseline | ZEUS | speedup | PSNR | reuse/recompute |
|---:|---:|---:|---:|---:|
| `375s` | `374s` | `1.003x` | `14.917 dB` | `4 / 12` branch calls |

Conclusion: only a smoke test; not used for quality/speed decisions.

### 4.2 Formal Fixed ZEUS, 10 Prompts

Result root: `experiment_results/wan22_zeus_timestep_cache_50step_45f_480p_full_20260608_114307`

Config:

| Item | Value |
|---|---|
| prompts | 10 prompts from `prompt.txt` |
| seed | fixed `42` |
| method | `--timestep_cache zeus` |
| acc range | `8 <= step < 47` |
| schedule | denominator `3`, modular `(0,1)` |
| caching mode | `reuse_interp` |
| max interval | `6` |
| lagrange | term `4`, int `4`, step `24` |

Aggregate result:

| prompts | baseline total | ZEUS total | speedup | mean PSNR |
|---:|---:|---:|---:|---:|
| `10` | `5262.025s` | `2649.240s` | `1.986x` | `23.705 dB` |

Prompt 01 detail:

| baseline | ZEUS | speedup | PSNR | reuse/recompute timesteps |
|---:|---:|---:|---:|---:|
| `522.872s` | `263.628s` | `1.983x` | `22.226 dB` | `25 / 25` |

Conclusion:

- Fixed-cadence ZEUS is the strongest completed multi-prompt result so far.
- It is much faster than conservative threshold cache while preserving mid-range PSNR.
- Prompt quality varies, but aggregate is usable as a baseline for later cache combinations.

## 5. ZEUS-threshold Timestep Cache

### 5.1 Prompt 01, 7-threshold Pilot

Result root: `experiment_results/wan22_zeus_threshold_prompt01_7th_20260608_162827`

Config:

| Item | Value |
|---|---|
| prompt | prompt 01 |
| seed | `42` |
| method | `zeus-threshold` |
| thresholds | `0.001`, `0.005`, `0.02`, `0.08`, `0.20`, `0.60`, `1.50` |
| mode | `reuse_interp` |

Result:

| threshold | speedup | mean PSNR | reuse/recompute timesteps |
|---:|---:|---:|---:|
| `0.001` | `1.000x` | `Infinity` | `0 / 50` |
| `0.005` | `1.111x` | `26.954 dB` | `5 / 45` |
| `0.02` | `1.607x` | `18.606 dB` | `19 / 31` |
| `0.08` | `2.251x` | `18.873 dB` | `28 / 22` |
| `0.20` | `2.599x` | `18.900 dB` | `31 / 19` |
| `0.60` | `2.734x` | `18.928 dB` | `32 / 18` |
| `1.50` | `2.731x` | `18.928 dB` | `32 / 18` |

Conclusion:

- `0.005` is the only high-quality prompt 01 threshold point.
- Starting at `0.02`, quality drops to roughly `18.6-18.9 dB`.
- Reuse saturates around `32/50` unique timesteps because of guards and schedule structure.

### 5.2 Prompt 01, Timestep-aware Interpolation

Result root: `experiment_results/wan22_zeus_threshold_taware_prompt01_5th_20260608_191714`

Config difference: same threshold cache, but interpolation uses timestep-aware scale between recent recompute step indices.

Result:

| threshold | speedup | mean PSNR | reuse/recompute timesteps |
|---:|---:|---:|---:|
| `0.005` | `1.110x` | `28.196 dB` | `5 / 45` |
| `0.02` | `1.618x` | `18.075 dB` | `19 / 31` |
| `0.08` | `2.263x` | `17.782 dB` | `28 / 22` |
| `0.20` | `2.620x` | `17.666 dB` | `31 / 19` |
| `0.60` | `2.764x` | `17.622 dB` | `32 / 18` |

Conclusion:

- Timestep-aware interpolation improves the conservative `0.005` point (`28.196 dB` vs `26.954 dB`).
- For larger thresholds it is worse than reuse_interp on prompt 01.
- Not selected for current three-cache grid; reuse_interp remains the default.

### 5.3 10-prompt ZEUS-threshold reuse_interp

Result root: `experiment_results/wan22_zeus_threshold_reuse_interp_10prompt_5th_20260608_195427`

Config:

| Item | Value |
|---|---|
| prompts | 10 prompts |
| thresholds | `0.005`, `0.02`, `0.08`, `0.20`, `0.60` |
| runner | single-process batch runner |
| mode | `reuse_interp` |

Aggregate result:

| threshold | speedup | mean PSNR | min PSNR | reuse/recompute timesteps |
|---:|---:|---:|---:|---:|
| `0.005` | `1.119x` | `26.208 dB` | `17.26 dB` | `53 / 447` |
| `0.02` | `1.576x` | `20.955 dB` | `14.18 dB` | `183 / 317` |
| `0.08` | `2.259x` | `20.932 dB` | `13.82 dB` | `280 / 220` |
| `0.20` | `2.614x` | `20.985 dB` | `13.88 dB` | `310 / 190` |
| `0.60` | `2.754x` | `21.020 dB` | `13.90 dB` | `320 / 180` |

Conclusion:

- On average, larger thresholds still report about `21 dB`, but min PSNR is poor.
- The high-quality threshold `0.005` has only modest speedup.
- Multi-prompt behavior is less stable than the formal fixed ZEUS cadence.

## 6. Block Cache Only

Result root: `experiment_results/wan22_block_cache_only_50step_45f_480p_20260609_125436`

Config:

| Item | Value |
|---|---|
| prompt | prompt 01 |
| seed | `42` |
| caches | timestep disabled, CFG disabled |
| methods | BWCache and block-group |
| BWCache thresholds | `0.05`, `0.15`, `0.30` |
| block-group thresholds | `0.01`, `0.03`, `0.05` |
| block-group settings | group size `5`, pooled relative-L1, start/end `0.1/0.9`, max reuse `3` |

Result:

| method | threshold | speedup | mean PSNR | reuse/recompute |
|---|---:|---:|---:|---:|
| BWCache | `0.05` | `1.030x` | `28.895 dB` | `4 / 96` |
| BWCache | `0.15` | `1.597x` | `18.370 dB` | `41 / 59` |
| BWCache | `0.30` | `1.674x` | `17.632 dB` | `44 / 56` |
| block-group | `0.01` | `0.982x` | `Infinity` | `0 / 800` |
| block-group | `0.03` | `1.361x` | `19.396 dB` | `244 / 556` |
| block-group | `0.05` | `1.713x` | `19.491 dB` | `374 / 426` |

Conclusion:

- BWCache `0.05` is high quality but almost no speedup.
- block-group is better for aggressive speed/quality tradeoff.
- `block-group 0.03~0.05` is useful for combined cache experiments, but quality is already around `19.4 dB` when used alone.

## 7. Three-cache Development Runs

These runs evaluated timestep `zeus-threshold=0.02`, block-group `0.03`, and CFG threshold `0.03` under evolving CFG policies.

### 7.1 Initial Three-cache Merge

Result root: `experiment_results/wan22_merge_three_cache_prompt01_20260609_153438`

Config:

| Cache | Setting |
|---|---|
| timestep | `zeus-threshold=0.02` |
| block-group | threshold `0.03`, group size `5`, start/end `0.1/0.9`, max reuse `3` |
| CFG | threshold `0.03`, old cond-output-oriented behavior, force uncond recompute on miss |

Result:

| compute | speedup | PSNR | timestep reuse/recompute | block reuse/recompute | CFG reuse/recompute |
|---:|---:|---:|---:|---:|---:|
| `357.403s` | `1.462x` | `18.001 dB` | `19 / 70` branch calls | `26 / 534` group calls | `11 / 39` |

Conclusion: useful OOM/behavior smoke, but quality too low.

### 7.2 Internal CFG Metric + Full Refresh

Result root: `experiment_results/wan22_merge_three_cache_internal_metric_prompt01_20260609_162556`

Change: CFG metric switched to `timestep_modulated_input_rel_l1`; CFG miss forced full refresh of both cond and uncond.

Result:

| compute | speedup | PSNR | delta vs initial |
|---:|---:|---:|---:|
| `492.713s` | `1.061x` | `19.623 dB` | `+1.621 dB`, `+135.310s` |

Conclusion: quality improved, but full refresh was too expensive.

### 7.3 Optimized CFG Scheduling

Result root: `experiment_results/wan22_merge_three_cache_optimized_schedule_prompt01_20260609_164735`

Change: when using input-feature CFG metric, decide CFG hit/miss before running cond, avoiding duplicate cond work on miss.

Result:

| compute | speedup | PSNR | delta vs internal metric |
|---:|---:|---:|---:|
| `381.459s` | `1.370x` | `19.603 dB` | `-111.254s`, `-0.019 dB` |

Conclusion:

- Scheduling optimization recovered most speed without losing quality.
- Later decision: disable CFG miss forced refresh for threshold grid because it is too costly and not always beneficial.

## 8. Cache Ablation

Result root: `experiment_results/wan22_cache_ablation_prompt01_50step_45f_480p_20260609_184625`

Config:

| Cache | Setting |
|---|---|
| timestep | `zeus-threshold=0.02`, `reuse_interp`, `acc_range=(8,47)` |
| block | block-group `0.03`, group size `5`, start/end `0.1/0.9`, max reuse `3` |
| CFG | threshold `0.03`, metric `timestep_modulated_input_rel_l1`, max reuse `3` |
| runner | single-process ablation runner |
| note | this ablation used the full-refresh CFG policy present at that time |

Result:

| candidate | caches | speedup | mean PSNR |
|---|---|---:|---:|
| `timestep_only` | timestep | `1.600x` | `18.606 dB` |
| `block_only` | block-group | `1.362x` | `19.396 dB` |
| `cfg_only` | CFG | `1.148x` | `21.571 dB` |
| `timestep_block` | timestep + block | `1.748x` | `18.159 dB` |
| `timestep_cfg` | timestep + CFG | `1.332x` | `20.910 dB` |
| `block_cfg` | block + CFG | `1.352x` | `19.446 dB` |
| `all_three` | timestep + block + CFG | `1.370x` | `19.603 dB` |

Conclusion:

- `timestep_only` here is `zeus-threshold=0.02`, not the fixed ZEUS formal run.
- CFG alone is not the main quality drop source.
- Timestep threshold `0.02` is the strongest source of quality loss.
- CFG can partially recover timestep-only quality, but at speed cost.

## 9. Three-cache Threshold Grid

Result root: `experiment_results/wan22_three_cache_threshold_grid_prompt01_50step_45f_480p_20260610_012518`

Config:

| Item | Value |
|---|---|
| prompt | prompt 01 |
| runner | single-process `WanT2V` pipeline, no repeated checkpoint load |
| total combinations | `64` |
| timestep thresholds | `0.001`, `0.005`, `0.02`, `0.60` |
| block-group thresholds | `0.001`, `0.015`, `0.03`, `1.00` |
| CFG thresholds | `0.001`, `0.02`, `0.03`, `1.00` |
| CFG force refresh | disabled (`False`) |

Completeness:

| Item | Count |
|---|---:|
| completed candidates | `64 / 64` |
| failed records | `0` |
| videos | `64` |
| PSNR JSON | `64` |
| candidate ffprobe JSON | `64` |
| baseline ffprobe JSON | `1` |

Key results:

| Selection | candidate | timestep | block | CFG | speedup | PSNR |
|---|---|---:|---:|---:|---:|---:|
| no-use control | `ts_0p001__bg_0p001__cfg_0p001` | `0.001` | `0.001` | `0.001` | `0.938x` | `Infinity` |
| fastest | `ts_0p6__bg_1__cfg_1` | `0.60` | `1.00` | `1.00` | `4.080x` | `15.225 dB` |
| best speed with PSNR >= 25 | `ts_0p005__bg_0p001__cfg_0p001` | `0.005` | `0.001` | `0.001` | `1.039x` | `26.954 dB` |
| best speed with PSNR >= 22 | `ts_0p005__bg_0p015__cfg_0p03` | `0.005` | `0.015` | `0.03` | `1.204x` | `23.448 dB` |
| best speed with PSNR >= 20 | `ts_0p005__bg_0p03__cfg_0p02` | `0.005` | `0.03` | `0.02` | `1.369x` | `20.042 dB` |
| best speed with PSNR >= 18 | `ts_0p6__bg_1__cfg_0p001` | `0.60` | `1.00` | `0.001` | `3.842x` | `18.128 dB` |

Selected rows:

| timestep | block | CFG | speedup | PSNR | Interpretation |
|---:|---:|---:|---:|---:|---|
| `0.001` | `0.001` | `0.02` | `1.027x` | `26.732 dB` | CFG-only conservative high quality |
| `0.001` | `0.001` | `0.03` | `1.121x` | `21.571 dB` | CFG-only more aggressive |
| `0.005` | `0.015` | `0.03` | `1.204x` | `23.448 dB` | best current >=22 dB point |
| `0.005` | `0.03` | `0.02` | `1.369x` | `20.042 dB` | fastest >=20 dB point |
| `0.02` | `0.03` | `0.02` | `1.624x` | `18.248 dB` | quality too low |
| `0.60` | `1.00` | `1.00` | `4.080x` | `15.225 dB` | speed upper bound only |

Conclusion:

- With CFG miss forced recompute disabled, speed ceiling increases substantially, but quality is threshold-sensitive.
- `timestep=0.005` is the best starting point for useful quality.
- `block=0.015` with `cfg=0.03` gives the current best `>=22 dB` tradeoff.
- `block=0.03` with `cfg=0.02` is the current fastest point near `20 dB`.
- `timestep=0.02` or `0.60` generally pushes quality below the likely useful range, unless the target accepts very low PSNR.

## 10. Failed, Superseded, or Less Informative Runs

| Experiment | Reason |
|---|---|
| `wan22_zeus_timestep_cache_50step_45f_480p_full_20260608_104151` | failed early baseline attempt; superseded |
| `wan22_zeus_timestep_cache_50step_45f_480p_full_20260608_104311` | stopped/interim after prompt 01 when PSNR/timing requirements were refined |
| `wan22_smoke_zeus_20260608_0951` | pipeline smoke only, 5 frames and 8 steps; not used for formal decisions |

## 11. Current Conclusions

1. Fixed ZEUS cadence remains the best validated multi-prompt result: `1.986x / 23.705 dB`.
2. Threshold timestep cache is sensitive. `0.005` preserves quality but gives modest speed; `0.02+` is risky.
3. Block-group cache is promising for speed but hurts quality around `0.03+`; more conservative thresholds between `0.015` and `0.03` are worth refining.
4. CFG cache alone can provide mild acceleration and moderate quality; it is not the primary source of degradation.
5. CFG miss forced full refresh improves quality in some cases but is too expensive; current experiments disable it.
6. Three-cache combinations can reach `4x` speedup, but useful quality regions are much narrower.
7. Current best prompt 01 candidate for follow-up is `timestep=0.005, block=0.015, cfg=0.03`, with `1.204x / 23.448 dB`.
8. A more aggressive follow-up candidate is `timestep=0.005, block=0.03, cfg=0.02`, with `1.369x / 20.042 dB`.

## 12. Recommended Next Experiments

### 12.1 Multi-prompt validation of top three-cache candidates

Run 10 prompts for:

| timestep | block | CFG | Reason |
|---:|---:|---:|---|
| `0.005` | `0.015` | `0.03` | best prompt 01 result above `22 dB` |
| `0.005` | `0.03` | `0.02` | fastest prompt 01 result near `20 dB` |
| `0.005` | `0.001` | `0.02` | conservative CFG + timestep reference |
| `0.005` | `0.001` | `0.001` | high-quality timestep-only-like control |

### 12.2 Finer block threshold sweep

For fixed `timestep=0.005`, sweep:

```text
block_group_threshold = 0.010, 0.015, 0.020, 0.025, 0.030
cfg_threshold = 0.02, 0.03
```

Purpose: find whether block `0.02~0.025` can improve speed without dropping PSNR below `22 dB`.

### 12.3 Compare fixed ZEUS cadence with block/CFG

The strongest multi-prompt baseline is fixed ZEUS. A useful next direction is combining fixed ZEUS cadence with conservative block/CFG thresholds, rather than only `zeus-threshold`.
