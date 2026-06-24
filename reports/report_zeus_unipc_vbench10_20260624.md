# ZEUS + UniPC on VBench10

## Summary

- Date: 2026-06-24
- Result root: `/hy-tmp/wan22_zeus_unipc_vbench10_50step_45f_480p_20260624_192306`
- Summary CSV: `/hy-tmp/wan22_zeus_unipc_vbench10_50step_45f_480p_20260624_192306/results/summary.csv`
- Aggregate JSON: `/hy-tmp/wan22_zeus_unipc_vbench10_50step_45f_480p_20260624_192306/results/aggregate.json`
- Prompt set: `VBench10`
- Completed pairs: `10`
- Total baseline compute elapsed: `5392.020s`
- Total ZEUS compute elapsed: `2706.871s`
- Overall speedup: `1.992x`
- Mean FFmpeg PSNR: `23.822 dB`
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
| vbench10_001 | 537.828 | 270.468 | 1.989x | 18.528 | 16.440 | 45 |
| vbench10_002 | 539.236 | 270.367 | 1.994x | 26.430 | 21.700 | 45 |
| vbench10_003 | 539.671 | 271.056 | 1.991x | 18.739 | 16.000 | 45 |
| vbench10_004 | 538.794 | 270.655 | 1.991x | 22.739 | 21.570 | 45 |
| vbench10_005 | 539.128 | 270.512 | 1.993x | 19.075 | 18.130 | 45 |
| vbench10_006 | 538.469 | 270.520 | 1.990x | 22.973 | 19.760 | 45 |
| vbench10_007 | 539.272 | 270.829 | 1.991x | 32.164 | 28.600 | 45 |
| vbench10_008 | 539.634 | 271.011 | 1.991x | 29.171 | 27.330 | 45 |
| vbench10_009 | 540.135 | 271.076 | 1.993x | 21.728 | 20.140 | 45 |
| vbench10_010 | 539.853 | 270.377 | 1.997x | 26.668 | 25.640 | 45 |

## Notes

- Speedup uses `inference_compute_elapsed_seconds`, excluding model loading and video saving.
- PSNR is computed with the FFmpeg `psnr` filter against the no-cache baseline generated with the same prompt, seed, shape, and solver.
- This run uses fixed ZEUS timestep caching only; block cache and CFG cache are disabled.
- The ZEUS interval is `[8,47]`, matching the official ZEUS Wan demo interval.
