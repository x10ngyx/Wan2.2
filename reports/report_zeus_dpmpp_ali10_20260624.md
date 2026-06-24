# ZEUS + DPM++ on ali-10

## Summary

- Date: 2026-06-24
- Result root: `/hy-tmp/wan22_zeus_timestep_cache_50step_45f_480p_full_20260608_114307`
- Summary CSV: `/hy-tmp/wan22_zeus_timestep_cache_50step_45f_480p_full_20260608_114307/results/summary.csv`
- Aggregate JSON: `/hy-tmp/wan22_zeus_timestep_cache_50step_45f_480p_full_20260608_114307/results/aggregate.json`
- Prompt set: `ali-10`
- Completed pairs: `10`
- Total baseline compute elapsed: `5262.025s`
- Total ZEUS compute elapsed: `2649.240s`
- Overall speedup: `1.986x`
- Mean FFmpeg PSNR: `23.705 dB`
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
| ffprobe json | 20 | 20 |
| PSNR json | 10 | 10 |
| failed files | 0 | 0 |

All completed videos were validated by ffprobe as `832x480`, `45` frames, `16 fps`, and `2.8125s` duration.

## Per-Sample Results

| sample_id | baseline_s | zeus_s | speedup | mean_psnr | min_psnr | frames |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ali_001 | 522.872 | 263.628 | 1.983x | 22.226 | 20.170 | 45 |
| ali_002 | 521.857 | 263.626 | 1.980x | 23.414 | 21.620 | 45 |
| ali_003 | 522.431 | 263.447 | 1.983x | 30.061 | 28.640 | 45 |
| ali_004 | 531.003 | 268.301 | 1.979x | 19.679 | 18.110 | 45 |
| ali_005 | 531.946 | 268.075 | 1.984x | 37.056 | 35.200 | 45 |
| ali_006 | 532.251 | 262.693 | 2.026x | 20.268 | 19.890 | 45 |
| ali_007 | 522.066 | 263.245 | 1.983x | 27.786 | 26.230 | 45 |
| ali_008 | 522.546 | 263.538 | 1.983x | 16.822 | 14.710 | 45 |
| ali_009 | 522.572 | 264.325 | 1.977x | 18.892 | 17.580 | 45 |
| ali_010 | 532.481 | 268.362 | 1.984x | 20.848 | 18.640 | 45 |

## Notes

- Speedup uses `inference_compute_elapsed_seconds`, excluding model loading and video saving.
- PSNR is computed with the FFmpeg `psnr` filter against the no-cache baseline generated with the same prompt, seed, shape, and solver.
- This run uses fixed ZEUS timestep caching only; block cache and CFG cache are disabled.
- The ZEUS interval is `[8,47]`, matching the official ZEUS Wan demo interval.
