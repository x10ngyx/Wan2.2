# 2026-06-30 MiniDiT Transformer Predictor Comprehensive Report

## Scope

- User requested a comprehensive report for the Transformer-architecture predictor.
- Required content:
  - architecture diagram and architecture parameter settings;
  - training parameter settings;
  - train/test loss curves;
  - real inference settings and performance, per prompt and aggregate.

## Files Added

- `reports/report_mini_dit_transformer_predictor_comprehensive_20260630.md`
- `reports/assets/mini_dit_cls_training_loss_curves.svg`

## Sources Used

- Architecture diagram:
  `reports/assets/mini_dit_cls_predictor_architecture.svg`
- Model implementation:
  `adaptive_threshold_predictor/models.py`
- Sample-split training output:
  `/hy-tmp/wan22_adaptive_threshold_mini_dit_cls_convpatch_3x12x8_d96_l2_bs128_20260629_214906`
- Row-split training output:
  `/hy-tmp/wan22_adaptive_threshold_mini_dit_cls_convpatch_rowsplit_packed_d96_l2_bs128_20260629_232659`
- Online inference result root:
  `/hy-tmp/wan22_adaptive_seacache_mini_dit_split_compare_50step_45f_480p_20260630_025328`

## Notes

- `matplotlib` was not available in the current environment, so the loss curve was generated directly as SVG from `epoch_metrics.csv`.
- The online summary table has `predictor_call_count=0` and blank predictor elapsed fields, so the report marks predictor overhead as unavailable and uses compute elapsed/speedup for online performance.
- The report includes all 24 online candidates and the aggregate table.

## Validation

- `git diff --check` passed for:
  - `reports/report_mini_dit_transformer_predictor_comprehensive_20260630.md`
  - `reports/assets/mini_dit_cls_training_loss_curves.svg`
  - `PROGRESS.md`
  - this session log.
