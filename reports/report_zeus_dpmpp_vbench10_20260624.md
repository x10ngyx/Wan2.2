# ZEUS + DPM++ on VBench10

## Summary

- Date: 2026-06-24
- Result root: `/hy-tmp/wan22_zeus_vbench10_50step_45f_480p_20260624_003030`
- Summary CSV: `/hy-tmp/wan22_zeus_vbench10_50step_45f_480p_20260624_003030/results/summary.csv`
- Aggregate CSV: `/hy-tmp/wan22_zeus_vbench10_50step_45f_480p_20260624_003030/results/aggregate_by_method.csv`
- Prompt set: `VBench10`
- Completed pairs: `10`
- Total baseline compute elapsed: `5413.092s`
- Total ZEUS compute elapsed: `2678.871s`
- Overall speedup: `2.021x`
- Mean FFmpeg PSNR: `23.996 dB`
- Failed records: `0`

## Configuration

| Field | Value |
| --- | --- |
| `task` | `t2v-A14B` |
| `checkpoint` | `/hy-tmp/models/Wan2.2-T2V-A14B` |
| `size` | `832*480` |
| `frame_num` | `45` |
| `sample_steps` | `50` |
| `sample_solver` | `dpm++` |
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
| fixed-ZEUS ffprobe json | 20 | 20 |
| fixed-ZEUS PSNR summary rows | 10 | 10 |
| failed files | 0 | 0 |

All completed videos were validated by ffprobe as `832x480`, `45` frames, `16 fps`, and `2.8125s` duration.

## Per-Sample Results

| sample_id | baseline_s | zeus_s | speedup | mean_psnr | min_psnr | frames |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vbench10_001 | 538.211 | 267.797 | 2.010x | 18.567 | 14.960 | 45 |
| vbench10_002 | 539.263 | 267.605 | 2.015x | 29.083 | 25.120 | 45 |
| vbench10_003 | 539.507 | 268.207 | 2.012x | 20.035 | 17.270 | 45 |
| vbench10_004 | 539.041 | 267.582 | 2.014x | 23.583 | 22.390 | 45 |
| vbench10_005 | 538.963 | 268.147 | 2.010x | 19.862 | 18.590 | 45 |
| vbench10_006 | 541.959 | 267.624 | 2.025x | 21.309 | 19.880 | 45 |
| vbench10_007 | 543.609 | 267.749 | 2.030x | 39.179 | 35.190 | 45 |
| vbench10_008 | 544.246 | 268.393 | 2.028x | 20.921 | 20.010 | 45 |
| vbench10_009 | 544.039 | 268.109 | 2.029x | 19.460 | 18.430 | 45 |
| vbench10_010 | 544.254 | 267.658 | 2.033x | 27.961 | 26.110 | 45 |

## Notes

- Speedup uses `inference_compute_elapsed_seconds`, excluding model loading and video saving.
- PSNR is computed with the FFmpeg `psnr` filter against the no-cache baseline generated with the same prompt, seed, shape, and solver.
- This run uses fixed ZEUS timestep caching only; block cache and CFG cache are disabled.
- The ZEUS interval is `[8,47]`, matching the official ZEUS Wan demo interval.
