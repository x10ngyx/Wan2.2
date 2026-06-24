# VBench10 Three-Cache Merge vs Timestep-Only Comparison Report

Generated: 2026-06-23 Asia/Shanghai.

This report compares the completed VBench10 three-cache SEA merge sweep against the completed timestep-only SeaCache sweep. Both experiments use the same 10 prompts, Wan2.2 T2V-A14B, seed 42, 45 frames, 50 sampling steps, `832*480`, and `dpm++`. The primary comparison axes are speedup over no-cache baseline and PSNR against the no-cache baseline.

Source files:

- three-cache aggregate: `/hy-tmp/wan22_three_cache_sea_vbench10_50step_45f_480p_20260619_3sea_merged_parent/merged/aggregate_by_candidate.csv`

- three-cache full table: `/hy-tmp/wan22_three_cache_sea_vbench10_50step_45f_480p_20260619_3sea_merged_parent/merged/summary.csv`

- timestep-only aggregate: `/hy-tmp/wan22_seacache_vbench10_50step_45f_480p_20260618_161845/merged/aggregate_by_threshold.csv`

- timestep-only full table: `/hy-tmp/wan22_seacache_vbench10_50step_45f_480p_20260618_161845/merged/summary.csv`


## 1. Setup Difference

| item | timestep-only | three-cache merge |
| --- | --- | --- |
| cache mechanisms | timestep SeaCache only | timestep SeaCache + block-group SEA cache + CFG SEA cache |
| threshold search | 10 one-dimensional thresholds: 0.10 to 0.80 | 64 combinations: timestep, block, CFG in {0.05, 0.10, 0.20, 0.50} |
| granularity | single threshold controls timestep reuse | separate controls for timestep reuse, block reuse, and CFG reuse |
| candidate count | 100 rows = 10 prompts x 10 thresholds | 640 rows = 10 prompts x 64 combinations |
| comparison baseline | no-cache generation for each prompt | no-cache generation for each prompt |


## 2. Headline Comparison

| case | method | setting | ts | bg | cfg | speedup | mean_psnr | min_psnr | t_reuse | bg_reuse | cfg_reuse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| merge fastest | three-cache merge | sea_ts_0p50__sea_bg_0p50__sea_cfg_0p50 | 0.50 | 0.50 | 0.50 | 3.524 | 20.058 | 14.770 | 390 | 432 | 356 |
| timestep fastest | timestep-only | th_0p80 | 0.80 | - | - | 3.535 | 19.657 | 14.200 | 390 |  |  |
| merge highest PSNR | three-cache merge | sea_ts_0p05__sea_bg_0p05__sea_cfg_0p10 | 0.05 | 0.05 | 0.10 | 0.978 | 37.499 | 29.110 | 0 | 0 | 56 |
| timestep highest PSNR | timestep-only | th_0p10 | 0.10 | - | - | 1.108 | 35.897 | 27.970 | 56 |  |  |
| merge best quality while >=1x | three-cache merge | sea_ts_0p05__sea_bg_0p10__sea_cfg_0p10 | 0.05 | 0.10 | 0.10 | 1.020 | 36.126 | 28.270 | 0 | 364 | 56 |
| timestep best quality while >=1x | timestep-only | th_0p10 | 0.10 | - | - | 1.108 | 35.897 | 27.970 | 56 |  |  |
| merge best quality while >=2x | three-cache merge | sea_ts_0p50__sea_bg_0p05__sea_cfg_0p05 | 0.50 | 0.05 | 0.05 | 2.417 | 20.679 | 13.690 | 346 | 0 | 0 |
| timestep best quality while >=2x | timestep-only | th_0p50 | 0.50 | - | - | 2.744 | 20.679 | 13.690 | 346 |  |  |


Key observations:

- Fastest points are similar: merge reaches `3.524`x, timestep-only reaches `3.535`x. The fastest merge point has slightly higher mean PSNR (`20.058` vs `19.657`), but both are aggressive low-quality settings.

- Merge improves the conservative/quality-preserving region: best merge setting while still faster than baseline is `sea_ts_0p05__sea_bg_0p10__sea_cfg_0p10` with `1.020`x and `36.126` PSNR, compared with timestep-only `th_0p10` at `1.108`x and `35.897` PSNR.

- At the high-quality end, merge can exceed timestep-only PSNR: best merge PSNR is `37.499`, best timestep-only PSNR is `35.897`. The best merge PSNR point is slightly slower than baseline, so it is mainly useful as a quality upper bound rather than a speed setting.


## 3. Best Quality at Fixed Minimum Speed

For each minimum speedup target, this table picks the highest-PSNR setting from each method. Positive `psnr_gain` means merge has better quality at that speed target.

| min_speed | merge_setting | merge_speed | merge_psnr | timestep_setting | timestep_speed | timestep_psnr | merge_psnr_gain |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0x | sea_ts_0p05__sea_bg_0p10__sea_cfg_0p10 | 1.020 | 36.126 | th_0p10 | 1.108 | 35.897 | 0.229 |
| 1.5x | sea_ts_0p20__sea_bg_0p05__sea_cfg_0p20 | 1.515 | 25.759 | th_0p20 | 1.574 | 25.958 | -0.199 |
| 2.0x | sea_ts_0p50__sea_bg_0p05__sea_cfg_0p05 | 2.417 | 20.679 | th_0p50 | 2.744 | 20.679 | 0.000 |
| 2.5x | sea_ts_0p20__sea_bg_0p50__sea_cfg_0p20 | 2.657 | 20.416 | th_0p50 | 2.744 | 20.679 | -0.263 |
| 3.0x | sea_ts_0p50__sea_bg_0p50__sea_cfg_0p05 | 3.190 | 20.116 | th_0p60 | 3.125 | 20.659 | -0.543 |
| 3.3x | sea_ts_0p50__sea_bg_0p50__sea_cfg_0p50 | 3.524 | 20.058 | th_0p70 | 3.339 | 19.838 | 0.220 |


## 4. Best Speed at Fixed Minimum PSNR

For each PSNR floor, this table picks the fastest setting from each method. Positive `speed_gain` means merge is faster at that quality floor.

| min_psnr | merge_setting | merge_speed | merge_psnr | timestep_setting | timestep_speed | timestep_psnr | merge_speed_gain |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 20 | sea_ts_0p50__sea_bg_0p50__sea_cfg_0p50 | 3.524 | 20.058 | th_0p60 | 3.125 | 20.659 | 0.399 |
| 25 | sea_ts_0p20__sea_bg_0p20__sea_cfg_0p20 | 1.520 | 25.509 | th_0p20 | 1.574 | 25.958 | -0.054 |
| 30 | sea_ts_0p10__sea_bg_0p05__sea_cfg_0p10 | 1.032 | 36.113 | th_0p10 | 1.108 | 35.897 | -0.075 |
| 35 | sea_ts_0p10__sea_bg_0p05__sea_cfg_0p10 | 1.032 | 36.113 | th_0p10 | 1.108 | 35.897 | -0.075 |
| 36 | sea_ts_0p10__sea_bg_0p05__sea_cfg_0p10 | 1.032 | 36.113 |  |  |  |  |


## 5. Matched Timestep Threshold Comparison

This table compares timestep-only thresholds against merge candidates with the same timestep threshold when that threshold exists in the merge grid. `merge_same_or_faster` is the highest-PSNR merge setting at least as fast as the timestep-only point. `merge_same_or_better_quality` is the fastest merge setting at least as high-quality as the timestep-only point.

| timestep_threshold | timestep_setting | timestep_speed | timestep_psnr | merge_same_or_faster | merge_speed | merge_psnr | psnr_gain | merge_same_or_better_quality | merge_quality_speed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.10 | th_0p10 | 1.108 | 35.897 | sea_ts_0p10__sea_bg_0p05__sea_cfg_0p20 | 1.221 | 26.387 | -9.510 | sea_ts_0p10__sea_bg_0p05__sea_cfg_0p10 | 1.032 |
| 0.15 | th_0p15 | 1.410 | 28.001 |  |  |  |  |  |  |
| 0.20 | th_0p20 | 1.574 | 25.958 | sea_ts_0p20__sea_bg_0p20__sea_cfg_0p50 | 1.943 | 20.725 | -5.233 | sea_ts_0p20__sea_bg_0p05__sea_cfg_0p05 | 1.439 |
| 0.25 | th_0p25 | 1.837 | 24.969 |  |  |  |  |  |  |
| 0.30 | th_0p30 | 1.979 | 23.515 |  |  |  |  |  |  |
| 0.40 | th_0p40 | 2.424 | 20.601 |  |  |  |  |  |  |
| 0.50 | th_0p50 | 2.744 | 20.679 | sea_ts_0p50__sea_bg_0p50__sea_cfg_0p05 | 3.190 | 20.116 | -0.563 | sea_ts_0p50__sea_bg_0p10__sea_cfg_0p05 | 2.418 |
| 0.60 | th_0p60 | 3.125 | 20.659 |  |  |  |  |  |  |
| 0.70 | th_0p70 | 3.339 | 19.838 |  |  |  |  |  |  |
| 0.80 | th_0p80 | 3.535 | 19.657 |  |  |  |  |  |  |


## 6. Pareto Frontiers

A point is on the frontier if no other setting from the same method is both faster and higher-PSNR.

| frontier | setting | ts | bg | cfg | speedup | mean_psnr | min_psnr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| three-cache merge | sea_ts_0p05__sea_bg_0p05__sea_cfg_0p10 | 0.05 | 0.05 | 0.10 | 0.978 | 37.499 | 29.110 |
| three-cache merge | sea_ts_0p05__sea_bg_0p10__sea_cfg_0p10 | 0.05 | 0.10 | 0.10 | 1.020 | 36.126 | 28.270 |
| three-cache merge | sea_ts_0p10__sea_bg_0p05__sea_cfg_0p10 | 0.10 | 0.05 | 0.10 | 1.032 | 36.113 | 28.450 |
| three-cache merge | sea_ts_0p05__sea_bg_0p05__sea_cfg_0p20 | 0.05 | 0.05 | 0.20 | 1.146 | 26.538 | 16.350 |
| three-cache merge | sea_ts_0p10__sea_bg_0p05__sea_cfg_0p20 | 0.10 | 0.05 | 0.20 | 1.221 | 26.387 | 16.470 |
| three-cache merge | sea_ts_0p20__sea_bg_0p05__sea_cfg_0p05 | 0.20 | 0.05 | 0.05 | 1.439 | 25.958 | 17.530 |
| three-cache merge | sea_ts_0p20__sea_bg_0p05__sea_cfg_0p20 | 0.20 | 0.05 | 0.20 | 1.515 | 25.759 | 17.290 |
| three-cache merge | sea_ts_0p20__sea_bg_0p20__sea_cfg_0p20 | 0.20 | 0.20 | 0.20 | 1.520 | 25.509 | 17.120 |
| three-cache merge | sea_ts_0p10__sea_bg_0p05__sea_cfg_0p50 | 0.10 | 0.05 | 0.50 | 1.527 | 20.770 | 13.080 |
| three-cache merge | sea_ts_0p10__sea_bg_0p20__sea_cfg_0p50 | 0.10 | 0.20 | 0.50 | 1.948 | 20.737 | 13.680 |
| three-cache merge | sea_ts_0p50__sea_bg_0p10__sea_cfg_0p05 | 0.50 | 0.10 | 0.05 | 2.418 | 20.679 | 13.690 |
| three-cache merge | sea_ts_0p20__sea_bg_0p50__sea_cfg_0p05 | 0.20 | 0.50 | 0.05 | 2.486 | 20.453 | 14.000 |
| three-cache merge | sea_ts_0p20__sea_bg_0p50__sea_cfg_0p20 | 0.20 | 0.50 | 0.20 | 2.657 | 20.416 | 13.750 |
| three-cache merge | sea_ts_0p50__sea_bg_0p50__sea_cfg_0p05 | 0.50 | 0.50 | 0.05 | 3.190 | 20.116 | 14.750 |
| three-cache merge | sea_ts_0p50__sea_bg_0p50__sea_cfg_0p50 | 0.50 | 0.50 | 0.50 | 3.524 | 20.058 | 14.770 |
| timestep-only | th_0p10 | 0.10 | - | - | 1.108 | 35.897 | 27.970 |
| timestep-only | th_0p15 | 0.15 | - | - | 1.410 | 28.001 | 18.520 |
| timestep-only | th_0p20 | 0.20 | - | - | 1.574 | 25.958 | 17.530 |
| timestep-only | th_0p25 | 0.25 | - | - | 1.837 | 24.969 | 18.230 |
| timestep-only | th_0p30 | 0.30 | - | - | 1.979 | 23.515 | 16.940 |
| timestep-only | th_0p50 | 0.50 | - | - | 2.744 | 20.679 | 13.690 |
| timestep-only | th_0p60 | 0.60 | - | - | 3.125 | 20.659 | 14.160 |
| timestep-only | th_0p70 | 0.70 | - | - | 3.339 | 19.838 | 14.460 |
| timestep-only | th_0p80 | 0.80 | - | - | 3.535 | 19.657 | 14.200 |


## 7. Merge Settings That Dominate Timestep-Only Points

Rows below are merge settings that are both at least as fast and at least as high-PSNR as one or more timestep-only aggregate points. This is the clearest evidence of merge advantage.

| setting | ts | bg | cfg | speedup | mean_psnr | min_psnr | t_reuse | bg_reuse | cfg_reuse | dominates_n | fastest_dominated_timestep | best_quality_dominated_timestep |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sea_ts_0p50__sea_bg_0p50__sea_cfg_0p50 | 0.50 | 0.50 | 0.50 | 3.524 | 20.058 | 14.770 | 390 | 432 | 356 | 1 | th_0p70 | th_0p70 |


## 8. Conclusion

- Merge expands the search space from 10 to 64 aggregate settings and produces `62` faster-than-baseline settings; `6` of them also keep mean PSNR >= 30.

- The main advantage of merge is not maximum speed. Maximum speed is roughly tied with timestep-only. The advantage is controllability: separate timestep, block, and CFG thresholds create settings that preserve more PSNR at comparable speed in the conservative region.

- The clearest practical merge setting is `sea_ts_0p05__sea_bg_0p10__sea_cfg_0p10`: it is faster than baseline (`1.020`x) and has high mean PSNR (`36.126`). This outperforms the best timestep-only faster-than-baseline quality point by about `0.229` dB mean PSNR, albeit with lower speedup.

- If the target is at least 2x speedup, merge and timestep-only converge to similar quality: merge `sea_ts_0p50__sea_bg_0p05__sea_cfg_0p05` gives `2.417`x / `20.679` PSNR, while timestep-only `th_0p50` gives `2.744`x / `20.679` PSNR. In this aggressive region, merge is not clearly better on PSNR, but it offers more combinations for selecting a speed/quality point.

- Based on these 10 prompts, merge is most useful when the goal is moderate acceleration with less quality loss, not when only the highest possible acceleration matters.
