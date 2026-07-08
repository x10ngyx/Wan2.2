# Session Log: Predictor Speedup Report

Date: 2026-07-06

## Work Done

- Added `reports/report_predictor_speedup.md`.
- Added `reports/make_predictor_speedup_training_curves.py`.
- Generated `reports/assets/predictor_speedup_training_loss_curves.svg`.
- Updated `PROGRESS.md` to mark the speedup-conditioned MiniDiT row-split online sweep as completed and to link the new report.
- Revised the training-loss SVG after review: the plot now shows only speedup-conditioned train/test loss curves, while original no-speedup loss remains in the report table as numeric comparison.
- Added the two condition-only ablations to `reports/report_predictor_speedup.md`: full condition (`timestep + target_psnr + target_speedup`) and speedup-only (`timestep + target_speedup`).

## Data Sources

- Original Transformer offline runs:
  - `/hy-tmp/wan22_adaptive_threshold_mini_dit_cls_convpatch_3x12x8_d96_l2_bs128_20260629_214906`
  - `/hy-tmp/wan22_adaptive_threshold_mini_dit_cls_convpatch_rowsplit_packed_d96_l2_bs128_20260629_232659`
- Speedup-conditioned Transformer offline runs:
  - `/hy-tmp/wan22_adaptive_threshold_mini_dit_cls_convpatch_3x12x8_d96_l2_bs128_speedup_20260706_171523`
  - `/hy-tmp/wan22_adaptive_threshold_mini_dit_cls_convpatch_rowsplit_packed_d96_l2_bs128_speedup_20260706_171523`
- Original and speedup-conditioned 5-feature gated MLP offline runs.
- Original Transformer online row-split aggregate:
  - `/hy-tmp/wan22_adaptive_seacache_mini_dit_split_compare_50step_45f_480p_20260630_025328/results/aggregate_by_dataset_model_target.csv`
- Speedup-conditioned Transformer online sweep aggregate:
  - `/hy-tmp/wan22_adaptive_seacache_mini_dit_rowsplit_speedup_sweep_50step_45f_480p_20260706_194715/results/aggregate_by_dataset_model_target.csv`
- Condition-only ablations:
  - `/hy-tmp/wan22_adaptive_threshold_condition_only_fullcond_rowsplit_long100_20260706_221900`
  - `/hy-tmp/wan22_adaptive_threshold_condition_only_speeduponly_rowsplit_long100_20260706_221900`

## Validation

- Regenerated the SVG plot with `python reports/make_predictor_speedup_training_curves.py`.
- Verified the speedup-conditioned online sweep has `36` summary rows and `12` aggregate rows; tmux session was no longer running and no failed CSV was present.
- Reviewed the rendered Markdown source and the generated SVG path.

## Notes

- No predictor code was changed in this session.
- The online comparison table uses row-split Transformer checkpoints only, because the speedup-conditioned online sweep did not run sample split.
