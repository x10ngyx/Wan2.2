# 2026-06-30 MiniDiT Split Compare Completed

## Scope

- Checked final status for the adaptive SeaCache MiniDiT row-split vs sample-split validation.
- Confirmed the queued 5-feature gated MLP validation started after MiniDiT finished.

## MiniDiT Result

- Result root:
  `/hy-tmp/wan22_adaptive_seacache_mini_dit_split_compare_50step_45f_480p_20260630_025328`
- Completed candidates: `24/24`.
- Failed records: `0`.
- Runner log ends with:
  `Completed experiment: /hy-tmp/wan22_adaptive_seacache_mini_dit_split_compare_50step_45f_480p_20260630_025328`
- Final candidate:
  `openvid100_train_openvid_005_row_split_target_28`
  - compute elapsed: `389.564s`
  - baseline elapsed: `536.112s`
  - speedup: `1.376x`
  - mean PSNR: `26.018`
  - reuse decisions: `28`
  - threshold mean: `0.156`

## Final Aggregate

| dataset | split | target | rows | speedup | mean PSNR | target error | mean threshold |
|---|---|---:|---:|---:|---:|---:|---:|
| vbench10 | sample_split | 22 | 3 | 2.113 | 16.737 | -5.263 | 0.359 |
| vbench10 | sample_split | 28 | 3 | 1.582 | 23.796 | -4.204 | 0.211 |
| vbench10 | row_split | 22 | 3 | 2.068 | 20.466 | -1.534 | 0.329 |
| vbench10 | row_split | 28 | 3 | 1.539 | 25.469 | -2.531 | 0.178 |
| openvid100_train | sample_split | 22 | 3 | 2.598 | 22.151 | +0.151 | 0.487 |
| openvid100_train | sample_split | 28 | 3 | 1.794 | 29.019 | +1.019 | 0.252 |
| openvid100_train | row_split | 22 | 3 | 2.447 | 23.007 | +1.007 | 0.484 |
| openvid100_train | row_split | 28 | 3 | 1.633 | 27.710 | -0.290 | 0.232 |

## Comparison Notes

- VBench10:
  - Row split is closer to target than sample split at both targets.
  - Absolute target error improves by `3.729 dB` at target 22 and `1.673 dB` at target 28.
  - Row split is slightly slower, by about `0.04x` speedup.
  - Both splits undershoot target PSNR; row split undershoots less.
- OpenVid train:
  - Sample split is closer at target 22.
  - Row split is closer at target 28.
  - Row split is slower by about `0.15x-0.16x` speedup.

## Gated MLP Follow-On Status

- tmux session:
  `wan22_adaptive_mlp_gated5_split_20260630_050727`
- Result root:
  `/hy-tmp/wan22_adaptive_seacache_mlp_gated_5feature_range_split_compare_50step_45f_480p_20260630_050727`
- Current status at check:
  - `1/24` rows completed.
  - `0` failures.
  - Running `vbench10_vbench10_001_sample_split_target_28`.

## Files Changed

- Updated `PROGRESS.md`.
- Added this session log.

## Validation

- Read final MiniDiT `summary.csv` and `aggregate_by_dataset_model_target.csv`.
- Checked `failed/`; no failure records were present.
- Checked tmux and GPU state; MiniDiT session exited and gated MLP session is active.
