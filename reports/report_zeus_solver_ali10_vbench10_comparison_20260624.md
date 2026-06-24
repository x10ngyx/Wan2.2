# ZEUS Solver Comparison on ali-10 and VBench10

## Compared Reports

| Prompt set | Solver | Report |
| --- | --- | --- |
| ali-10 | `dpm++` | `reports/report_zeus_dpmpp_ali10_20260624.md` |
| ali-10 | `unipc` | `reports/report_zeus_unipc_ali10_20260624.md` |
| VBench10 | `dpm++` | `reports/report_zeus_dpmpp_vbench10_20260624.md` |
| VBench10 | `unipc` | `reports/report_zeus_unipc_vbench10_20260624.md` |

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
| `block_cache` | `none` |
| `cfg_cache` | `none` |

## Aggregate Results

| Prompt set | Solver | pairs | total_baseline_s | total_zeus_s | overall_speedup | mean_psnr |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ali-10 | `dpm++` | 10 | 5262.025 | 2649.240 | 1.986x | 23.705 |
| ali-10 | `unipc` | 10 | 5444.462 | 2731.038 | 1.994x | 24.631 |
| VBench10 | `dpm++` | 10 | 5413.092 | 2678.871 | 2.021x | 23.996 |
| VBench10 | `unipc` | 10 | 5392.020 | 2706.871 | 1.992x | 23.822 |

## ali-10 Per-Sample Comparison

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

## VBench10 Per-Sample Comparison

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
