# Session Log: Transformer vs 5-Feature Predictor Readout

Date: 2026-06-30 12:53 CST

## What I Did

- Read the MiniDiT Transformer comprehensive report, the 5-feature gated MLP
  comprehensive report, and the dedicated comparison report.
- Cross-checked the online `summary.csv` files for both predictors.
- Recomputed simple aggregate online metrics using Python standard-library CSV
  parsing because the default Python environment did not have pandas installed.

## Files Read

- `reports/report_mini_dit_transformer_predictor_comprehensive_20260630.md`
- `reports/report_gated_multifeature_mlp_predictor_comprehensive_20260630.md`
- `reports/report_adaptive_predictor_mini_dit_vs_gated_mlp_comparison_20260630.md`
- `/hy-tmp/wan22_adaptive_seacache_mini_dit_split_compare_50step_45f_480p_20260630_025328/results/summary.csv`
- `/hy-tmp/wan22_adaptive_seacache_mlp_gated_5feature_range_split_compare_50step_45f_480p_20260630_050727/results/summary.csv`

## Main Findings

- MiniDiT is much stronger than the 5-feature MLP on row-split offline MAE, but
  sample-split MAE is essentially tied.
- Online inference does not show a clean winner. The 5-feature MLP is slightly
  faster on average, while MiniDiT has slightly higher average PSNR.
- Both predictors still have weak PSNR target control, with only `8/24`
  candidates reaching the requested target in the current online comparison.
- The result supports the current diagnosis that the offline `candidate_inverse`
  training objective is not sufficiently aligned with online adaptive threshold
  control.

## Changes

- Appended a progress note to `PROGRESS.md`.
- Added this session log.

## Validation

- No code changes or experiments were run.
- No commit was made.
