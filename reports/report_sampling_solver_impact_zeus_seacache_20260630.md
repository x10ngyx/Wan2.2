# Sampling Solver Impact on ZEUS and SeaCache

## Data Coverage

| Cache method | Prompt set | `dpm++` coverage | `unipc` coverage | Included |
| --- | --- | --- | --- | --- |
| ZEUS | ali-10 | 10/10 | 10/10 | yes |
| ZEUS | VBench10 | 10/10 | 10/10 | yes |
| SeaCache | VBench10 | 10/10 | 10/10 | yes |
| SeaCache | ali-10 | prompt 1/2 pilot only | 10/10 | no |

## Shared Configuration

| Field | Value |
| --- | --- |
| `task` | `t2v-A14B` |
| `checkpoint` | `/hy-tmp/models/Wan2.2-T2V-A14B` |
| `size` | `832*480` |
| `frame_num` | `45` |
| `sample_steps` | `50` |
| `base_seed` | `42` |
| `sample_shift` | `12.0` |
| `sample_guide_scale` | `(3.0, 4.0)` |
| `block_cache` | `none` |
| `cfg_cache` | `none` |
| speed metric | `inference_compute_elapsed_seconds` |
| quality metric | FFmpeg PSNR against same-solver no-cache baseline |

## Source Artifacts

| Cache method | Prompt set | Solver | Source |
| --- | --- | --- | --- |
| ZEUS | ali-10 | `dpm++` | `reports/report_zeus_dpmpp_ali10_20260624.md` |
| ZEUS | ali-10 | `unipc` | `reports/report_zeus_unipc_ali10_20260624.md` |
| ZEUS | VBench10 | `dpm++` | `reports/report_zeus_dpmpp_vbench10_20260624.md` |
| ZEUS | VBench10 | `unipc` | `reports/report_zeus_unipc_vbench10_20260624.md` |
| SeaCache | VBench10 | `dpm++` summary | `/hy-tmp/wan22_vbench10_three_cache_full_merge_and_timestep_only_full_20260623/timestep_only_seacache_vbench10_full/wan22_seacache_vbench10_50step_45f_480p_20260618_161845/merged/summary.csv` |
| SeaCache | VBench10 | `dpm++` aggregate | `/hy-tmp/wan22_vbench10_three_cache_full_merge_and_timestep_only_full_20260623/timestep_only_seacache_vbench10_full/wan22_seacache_vbench10_50step_45f_480p_20260618_161845/merged/aggregate_by_threshold.csv` |
| SeaCache | VBench10 | `unipc` summary | `/hy-tmp/wan22_seacache_unipc_vbench10_50step_45f_480p_20260627_023222/results/summary.csv` |
| SeaCache | VBench10 | `unipc` aggregate | `/hy-tmp/wan22_seacache_unipc_vbench10_50step_45f_480p_20260627_023222/results/aggregate_by_threshold.csv` |

# ZEUS

## ZEUS Configuration

| Field | Value |
| --- | --- |
| `timestep_cache` | `zeus` |
| `zeus_acc_start` | `8` |
| `zeus_acc_end` | `47` |
| `zeus_denominator` | `3` |
| `zeus_modular` | `[0, 1]` |
| `zeus_caching_mode` | `reuse_interp` |
| `zeus_max_interval` | `6` |
| `zeus_lagrange_int` | `4` |
| `zeus_lagrange_step` | `24` |
| `zeus_lagrange_term` | `4` |

## ZEUS Aggregate Results

| Prompt set | Solver | pairs | total_baseline_s | total_zeus_s | overall_speedup | mean_psnr |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ali-10 | `dpm++` | 10 | 5262.025 | 2649.240 | 1.986x | 23.705 |
| ali-10 | `unipc` | 10 | 5444.462 | 2731.038 | 1.994x | 24.631 |
| VBench10 | `dpm++` | 10 | 5413.092 | 2678.871 | 2.021x | 23.996 |
| VBench10 | `unipc` | 10 | 5392.020 | 2706.871 | 1.992x | 23.822 |

## ZEUS ali-10 Per-Sample Comparison

| sample_id | dpm++ speedup | unipc speedup | dpm++ mean_psnr | unipc mean_psnr | psnr_delta_unipc_minus_dpmpp |
| --- | ---: | ---: | ---: | ---: | ---: |
| ali_001 | 1.983x | 1.988x | 22.226 | 21.408 | -0.818 |
| ali_002 | 1.980x | 1.990x | 23.414 | 29.477 | 6.063 |
| ali_003 | 1.983x | 1.996x | 30.061 | 28.943 | -1.118 |
| ali_004 | 1.979x | 1.994x | 19.679 | 21.788 | 2.109 |
| ali_005 | 1.984x | 1.997x | 37.056 | 36.726 | -0.330 |
| ali_006 | 2.026x | 1.994x | 20.268 | 20.404 | 0.136 |
| ali_007 | 1.983x | 1.992x | 27.786 | 28.522 | 0.736 |
| ali_008 | 1.983x | 1.996x | 16.822 | 19.406 | 2.584 |
| ali_009 | 1.977x | 1.994x | 18.892 | 19.324 | 0.432 |
| ali_010 | 1.984x | 1.993x | 20.848 | 20.313 | -0.535 |

## ZEUS VBench10 Per-Sample Comparison

| sample_id | dpm++ speedup | unipc speedup | dpm++ mean_psnr | unipc mean_psnr | psnr_delta_unipc_minus_dpmpp |
| --- | ---: | ---: | ---: | ---: | ---: |
| vbench10_001 | 2.010x | 1.989x | 18.567 | 18.528 | -0.039 |
| vbench10_002 | 2.015x | 1.994x | 29.083 | 26.430 | -2.653 |
| vbench10_003 | 2.012x | 1.991x | 20.035 | 18.739 | -1.296 |
| vbench10_004 | 2.014x | 1.991x | 23.583 | 22.739 | -0.844 |
| vbench10_005 | 2.010x | 1.993x | 19.862 | 19.075 | -0.787 |
| vbench10_006 | 2.025x | 1.990x | 21.309 | 22.973 | 1.664 |
| vbench10_007 | 2.030x | 1.991x | 39.179 | 32.164 | -7.015 |
| vbench10_008 | 2.028x | 1.991x | 20.921 | 29.171 | 8.250 |
| vbench10_009 | 2.029x | 1.993x | 19.460 | 21.728 | 2.268 |
| vbench10_010 | 2.033x | 1.997x | 27.961 | 26.668 | -1.293 |

# SeaCache

## SeaCache Configuration

| Field | Value |
| --- | --- |
| `timestep_cache` | `seacache` |
| thresholds compared | `0.10`, `0.20`, `0.30`, `0.50` |
| `seacache_power_exp` | `3.0` |
| `seacache_norm_mode` | `mean` |
| comparison scope | VBench10 |

## SeaCache VBench10 Aggregate Results

| threshold | dpm++ pairs | dpm++ speedup | dpm++ mean_psnr | dpm++ min_psnr | unipc pairs | unipc speedup | unipc mean_psnr | unipc min_psnr | mean_psnr_delta_unipc_minus_dpmpp |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.10 | 10 | 1.108x | 35.897 | 27.97 | 10 | 1.109x | 36.415 | 28.56 | 0.517 |
| 0.20 | 10 | 1.574x | 25.958 | 17.53 | 10 | 1.575x | 25.721 | 16.40 | -0.238 |
| 0.30 | 10 | 1.979x | 23.515 | 16.94 | 10 | 1.937x | 23.034 | 16.20 | -0.481 |
| 0.50 | 10 | 2.744x | 20.679 | 13.69 | 10 | 2.663x | 19.605 | 13.27 | -1.073 |

## SeaCache VBench10 Per-Sample Comparison, Threshold 0.10

| sample_id | dpm++ speedup | unipc speedup | dpm++ mean_psnr | unipc mean_psnr | psnr_delta_unipc_minus_dpmpp |
| --- | ---: | ---: | ---: | ---: | ---: |
| vbench10_001 | 1.115x | 1.117x | 32.608 | 32.890 | 0.282 |
| vbench10_002 | 1.117x | 1.117x | 42.542 | 42.356 | -0.185 |
| vbench10_003 | 1.095x | 1.095x | 31.720 | 31.046 | -0.674 |
| vbench10_004 | 1.118x | 1.119x | 36.091 | 33.772 | -2.320 |
| vbench10_005 | 1.095x | 1.096x | 30.030 | 30.568 | 0.538 |
| vbench10_006 | 1.092x | 1.095x | 35.110 | 36.432 | 1.322 |
| vbench10_007 | 1.118x | 1.118x | 46.819 | 47.378 | 0.559 |
| vbench10_008 | 1.118x | 1.120x | 31.390 | 35.216 | 3.826 |
| vbench10_009 | 1.095x | 1.096x | 32.880 | 33.903 | 1.023 |
| vbench10_010 | 1.118x | 1.118x | 39.783 | 40.586 | 0.803 |

## SeaCache VBench10 Per-Sample Comparison, Threshold 0.20

| sample_id | dpm++ speedup | unipc speedup | dpm++ mean_psnr | unipc mean_psnr | psnr_delta_unipc_minus_dpmpp |
| --- | ---: | ---: | ---: | ---: | ---: |
| vbench10_001 | 1.575x | 1.574x | 20.295 | 20.461 | 0.166 |
| vbench10_002 | 1.573x | 1.574x | 32.270 | 27.227 | -5.044 |
| vbench10_003 | 1.576x | 1.575x | 20.250 | 22.410 | 2.160 |
| vbench10_004 | 1.575x | 1.576x | 24.736 | 24.028 | -0.709 |
| vbench10_005 | 1.575x | 1.570x | 21.166 | 21.196 | 0.030 |
| vbench10_006 | 1.570x | 1.577x | 24.488 | 24.796 | 0.308 |
| vbench10_007 | 1.575x | 1.575x | 41.091 | 37.598 | -3.492 |
| vbench10_008 | 1.575x | 1.577x | 24.702 | 27.992 | 3.289 |
| vbench10_009 | 1.576x | 1.575x | 20.576 | 23.872 | 3.296 |
| vbench10_010 | 1.575x | 1.575x | 30.007 | 27.626 | -2.381 |

## SeaCache VBench10 Per-Sample Comparison, Threshold 0.30

| sample_id | dpm++ speedup | unipc speedup | dpm++ mean_psnr | unipc mean_psnr | psnr_delta_unipc_minus_dpmpp |
| --- | ---: | ---: | ---: | ---: | ---: |
| vbench10_001 | 1.978x | 1.977x | 20.156 | 19.294 | -0.862 |
| vbench10_002 | 1.982x | 1.909x | 27.949 | 25.085 | -2.864 |
| vbench10_003 | 1.978x | 1.909x | 18.548 | 20.242 | 1.694 |
| vbench10_004 | 1.976x | 1.982x | 23.604 | 22.784 | -0.820 |
| vbench10_005 | 1.978x | 1.911x | 19.056 | 17.180 | -1.876 |
| vbench10_006 | 1.976x | 1.911x | 22.090 | 21.107 | -0.983 |
| vbench10_007 | 1.980x | 1.982x | 37.090 | 35.617 | -1.473 |
| vbench10_008 | 1.982x | 1.975x | 21.400 | 24.512 | 3.113 |
| vbench10_009 | 1.979x | 1.911x | 18.104 | 19.807 | 1.703 |
| vbench10_010 | 1.981x | 1.908x | 27.157 | 24.715 | -2.442 |

## SeaCache VBench10 Per-Sample Comparison, Threshold 0.50

| sample_id | dpm++ speedup | unipc speedup | dpm++ mean_psnr | unipc mean_psnr | psnr_delta_unipc_minus_dpmpp |
| --- | ---: | ---: | ---: | ---: | ---: |
| vbench10_001 | 2.796x | 2.658x | 15.582 | 15.376 | -0.206 |
| vbench10_002 | 2.804x | 2.662x | 19.977 | 19.394 | -0.583 |
| vbench10_003 | 2.665x | 2.661x | 16.490 | 15.446 | -1.044 |
| vbench10_004 | 2.798x | 2.666x | 22.051 | 20.851 | -1.200 |
| vbench10_005 | 2.662x | 2.666x | 18.471 | 17.512 | -0.959 |
| vbench10_006 | 2.655x | 2.663x | 18.148 | 17.095 | -1.053 |
| vbench10_007 | 2.803x | 2.664x | 33.678 | 28.607 | -5.071 |
| vbench10_008 | 2.805x | 2.663x | 18.941 | 22.457 | 3.516 |
| vbench10_009 | 2.665x | 2.665x | 18.963 | 15.372 | -3.591 |
| vbench10_010 | 2.805x | 2.663x | 24.487 | 23.943 | -0.544 |
