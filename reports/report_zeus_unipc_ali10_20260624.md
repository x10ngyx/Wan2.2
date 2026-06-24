# ZEUS + UniPC on ali-10

## Summary

- Date: 2026-06-24
- Result root: `/hy-tmp/wan22_zeus_unipc_ali10_50step_45f_480p_20260624_195011`
- Summary CSV: `/hy-tmp/wan22_zeus_unipc_ali10_50step_45f_480p_20260624_195011/results/summary.csv`
- Aggregate JSON: `/hy-tmp/wan22_zeus_unipc_ali10_50step_45f_480p_20260624_195011/results/aggregate.json`
- Prompt set: `ali-10`
- Completed pairs: `10`
- Total baseline compute elapsed: `5444.462s`
- Total ZEUS compute elapsed: `2731.038s`
- Overall speedup: `1.994x`
- Mean FFmpeg PSNR: `24.631 dB`
- Failed records: `0`

## Configuration

| Field | Value |
| --- | --- |
| `task` | `t2v-A14B` |
| `checkpoint` | `/hy-tmp/models/Wan2.2-T2V-A14B` |
| `size` | `832*480` |
| `frame_num` | `45` |
| `sample_steps` | `50` |
| `sample_solver` | `unipc` |
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

## Artifact Checks

| Artifact | Count | Expected |
| --- | ---: | ---: |
| baseline mp4 | 10 | 10 |
| ZEUS mp4 | 10 | 10 |
| ffprobe json | 20 | 20 |
| PSNR json | 10 | 10 |
| failed files | 0 | 0 |

All completed videos were validated by ffprobe as `832x480`, `45` frames, `16 fps`, and `2.8125s` duration.

## Per-Sample Results

| sample_id | baseline_s | zeus_s | speedup | mean_psnr | min_psnr | frames |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ali_001 | 542.826 | 273.062 | 1.988x | 21.408 | 20.200 | 45 |
| ali_002 | 543.820 | 273.216 | 1.990x | 29.477 | 26.270 | 45 |
| ali_003 | 544.568 | 272.889 | 1.996x | 28.943 | 27.550 | 45 |
| ali_004 | 545.097 | 273.318 | 1.994x | 21.788 | 20.050 | 45 |
| ali_005 | 544.567 | 272.635 | 1.997x | 36.726 | 35.300 | 45 |
| ali_006 | 544.694 | 273.190 | 1.994x | 20.404 | 19.880 | 45 |
| ali_007 | 544.809 | 273.472 | 1.992x | 28.522 | 26.180 | 45 |
| ali_008 | 545.298 | 273.161 | 1.996x | 19.406 | 14.460 | 45 |
| ali_009 | 544.300 | 272.958 | 1.994x | 19.324 | 18.080 | 45 |
| ali_010 | 544.483 | 273.137 | 1.993x | 20.313 | 18.530 | 45 |

## Notes

- Speedup uses `inference_compute_elapsed_seconds`, excluding model loading and video saving.
- PSNR is computed with the FFmpeg `psnr` filter against the no-cache baseline generated with the same prompt, seed, shape, and solver.
- This run uses fixed ZEUS timestep caching only; block cache and CFG cache are disabled.
- The ZEUS interval is `[8,47]`, matching the official ZEUS Wan demo interval.
