# 2026-06-16 Adaptive Predictor Report Ablation Tables

## Summary

- Added ablation result tables to `reports/report_adaptive_predictor.md`.
- No training or evaluation was rerun.

## Sources

- `/hy-tmp/wan22_adaptive_threshold_feature_ablation_cached_20260616_012409/feature_ablation_best_summary.csv`
- `/hy-tmp/wan22_adaptive_threshold_controls_20260616/control_summary.csv`
- `/hy-tmp/wan22_adaptive_threshold_grid_ablation_20260616_020314/grid_feature_ablation_best_summary.csv`

## Tables Added

- Feature training summary:
  - five real latent feature inputs at `2x2x2`;
  - `noise_feature` control;
  - `condition_only` blank/no-latent control.
- Pooling-size training summary:
  - `2x2x2`;
  - `2x4x4`;
  - `3x4x4`;
  - `4x4x4`;
  - all five feature sets where available.

## Columns

- parameter count;
- best epoch;
- best train loss;
- best validation loss;
- best validation MAE;
- last validation loss;
- last validation MAE.

## Files Changed

- `reports/report_adaptive_predictor.md`
- `PROGRESS.md`
- `logs/2026-06-16_adaptive_predictor_report_ablation_tables.md`
