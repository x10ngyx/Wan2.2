# 2026-06-30 MiniDiT Split Compare Progress Check

## Scope

- Checked progress for the adaptive SeaCache MiniDiT row-split vs sample-split online validation.
- User request was to inspect progress, not to modify runner behavior.

## Commands/Checks

- Read `PROGRESS.md`.
- Checked tmux sessions with `tmux ls`.
- Checked GPU activity with `nvidia-smi`.
- Inspected:
  - `/hy-tmp/wan22_adaptive_seacache_mini_dit_split_compare_50step_45f_480p_20260630_025328/results/summary.csv`
  - `/hy-tmp/wan22_adaptive_seacache_mini_dit_split_compare_50step_45f_480p_20260630_025328/results/aggregate_by_dataset_model_target.csv`
  - `/hy-tmp/wan22_adaptive_seacache_mini_dit_split_compare_50step_45f_480p_20260630_025328/logs/runner.log`
  - tmux pane for `wan22_adaptive_mini_dit_split_20260630_025328`

## Status

- MiniDiT tmux session is still running:
  `wan22_adaptive_mini_dit_split_20260630_025328`.
- Result root:
  `/hy-tmp/wan22_adaptive_seacache_mini_dit_split_compare_50step_45f_480p_20260630_025328`
- Completed rows: `23/24`.
- Failed records: `0`.
- Last running candidate:
  `openvid100_train_openvid_005_row_split_target_28`.
- At 2026-06-30 05:15 CST, tmux pane showed this last candidate at `17/50` sampling steps.
- GPU check showed A100 active with about `47317/81920 MiB` used and `100%` utilization.
- The queued 5-feature gated MLP session remains waiting for this MiniDiT session:
  `wan22_adaptive_mlp_gated5_split_20260630_050727`.

## Current Aggregate Snapshot

`results/aggregate_by_dataset_model_target.csv` currently has complete groups except `openvid100_train,row_split,target28`, which has `2/3` rows.

| dataset | split | target | rows | speedup | mean PSNR | target error |
|---|---|---:|---:|---:|---:|---:|
| vbench10 | sample_split | 22 | 3 | 2.113 | 16.737 | -5.263 |
| vbench10 | sample_split | 28 | 3 | 1.582 | 23.796 | -4.204 |
| vbench10 | row_split | 22 | 3 | 2.068 | 20.466 | -1.534 |
| vbench10 | row_split | 28 | 3 | 1.539 | 25.469 | -2.531 |
| openvid100_train | sample_split | 22 | 3 | 2.598 | 22.151 | +0.151 |
| openvid100_train | sample_split | 28 | 3 | 1.794 | 29.019 | +1.019 |
| openvid100_train | row_split | 22 | 3 | 2.447 | 23.007 | +1.007 |
| openvid100_train | row_split | 28 | 2 | 1.801 | 28.556 | +0.556 |

## Files Changed

- Updated `PROGRESS.md`.
- Added this session log.

## Next Step

- Recheck after the MiniDiT tmux exits; if `24/24` completes with no failures, record the final aggregate and summarize row split vs sample split.
