# 2026-06-23 Adaptive Predictor Training Curves Report

## Summary

- Added plotting script for adaptive threshold predictor training curves:
  - `experiments/adaptive_threshold_predictor/plot_training_curves.py`
- Generated plots and summary tables under:
  - `reports/assets/adaptive_training_curves/`
- Added combined report:
  - `reports/report_adaptive_predictor_training_curves.md`
  - The report now includes experiment task definition, model architecture, feature definitions, grid/parameter table, training hyperparameters, control baseline setup, all generated figures, quantitative tables, and conclusions.

## Inputs

- 3-epoch `2x2x2` feature ablation:
  - `/hy-tmp/wan22_adaptive_threshold_feature_ablation_cached_20260616_012409`
- 3-epoch larger-grid ablation:
  - `/hy-tmp/wan22_adaptive_threshold_grid_ablation_20260616_020314`
- Control baselines:
  - `/hy-tmp/wan22_adaptive_threshold_controls_20260616`
- 30-epoch convergence checks:
  - `/hy-tmp/wan22_adaptive_threshold_feature_ablation_long_20260616`
  - `/hy-tmp/wan22_adaptive_threshold_feature_ablation_hdim8_20260616`
  - `/hy-tmp/wan22_adaptive_threshold_feature_ablation_hdim16_20260616`

## Key Findings

- Best 3-epoch feature/grid setting remains `2x2x2 temporal_mean`, best validation loss `0.012259`.
- Larger pooling grids did not improve global best validation loss.
- No-feature/noise-feature controls have validation loss about `0.01465`, worse than `temporal_mean` and `latent_pool`.
- 30-epoch runs show early validation optimum and later overfitting, so current training should be described as early-stopping comparison rather than full convergence.
- `hdim16 temporal_mean` reaches similar best validation loss to hdim64 with far fewer parameters.

## Validation

- Ran:
  - `/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m experiments.adaptive_threshold_predictor.plot_training_curves --out_dir reports/assets/adaptive_training_curves`
  - `/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m py_compile experiments/adaptive_threshold_predictor/plot_training_curves.py`

## PDF Export

- Added PDF export helper:
  - `reports/export_adaptive_predictor_training_curves_pdf.py`
- Exported:
  - `reports/report_adaptive_predictor_training_curves.pdf`
- The host did not have pandoc, wkhtmltopdf, Chromium, WeasyPrint, or a system CJK font available.
- Downloaded Noto Sans CJK SC Regular to `/hy-tmp/fonts/NotoSansCJKsc-Regular.otf` for local PDF rendering. The font file is outside the repository.
- Validation:
  - `file reports/report_adaptive_predictor_training_curves.pdf` reported `PDF document, version 1.4, 6 pages`.
