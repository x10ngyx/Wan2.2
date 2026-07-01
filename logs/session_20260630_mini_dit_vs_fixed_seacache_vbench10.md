# 2026-06-30 MiniDiT Row-Split vs Fixed SeaCache VBench10

## Scope

- Compared the completed MiniDiT adaptive SeaCache row-split VBench10 results
  against normal fixed-threshold SeaCache on the same tested prompts.

## Sources

- MiniDiT row-split:
  `/hy-tmp/wan22_adaptive_seacache_mini_dit_split_compare_50step_45f_480p_20260630_025328/results/summary.csv`
- Fixed SeaCache dpm++:
  `/hy-tmp/wan22_vbench10_three_cache_full_merge_and_timestep_only_full_20260623/timestep_only_seacache_vbench10_full/wan22_seacache_vbench10_50step_45f_480p_20260618_161845/merged/summary.csv`

## Same-Prompt Fixed SeaCache Aggregate

Prompts: `vbench10_001`, `vbench10_002`, `vbench10_003`.

| threshold | speedup | mean PSNR |
|---:|---:|---:|
| 0.10 | 1.109 | 35.623 |
| 0.15 | 1.410 | 27.728 |
| 0.20 | 1.575 | 24.272 |
| 0.25 | 1.844 | 23.614 |
| 0.30 | 1.979 | 22.218 |
| 0.40 | 2.425 | 17.364 |
| 0.50 | 2.753 | 17.350 |
| 0.60 | 3.125 | 16.618 |
| 0.70 | 3.386 | 16.756 |
| 0.80 | 3.534 | 16.539 |

## Target Comparison

| target | method | threshold | speedup | mean PSNR | target error |
|---:|---|---:|---:|---:|---:|
| 22 | MiniDiT row split | adaptive | 2.068 | 20.466 | -1.534 |
| 22 | fixed SeaCache best by target closeness | 0.30 | 1.979 | 22.218 | +0.218 |
| 28 | MiniDiT row split | adaptive | 1.539 | 25.469 | -2.531 |
| 28 | fixed SeaCache best by target closeness | 0.15 | 1.410 | 27.728 | -0.272 |

## Takeaway

- Fixed SeaCache is closer to target PSNR on both target groups.
- MiniDiT row split is slightly faster than the closest fixed-threshold SeaCache point:
  `+0.089x` for target 22 and `+0.129x` for target 28.
- The speed gain is not enough to offset the larger PSNR undershoot on these VBench10 prompts.
