# Session Log: 5-Feature Gated MLP Predictor Comprehensive Report

Date: 2026-06-30

## Request

Write a comprehensive report for the 5-feature gated MLP adaptive threshold
predictor, following the format of:

```text
reports/report_mini_dit_transformer_predictor_comprehensive_20260630.md
```

Required contents:

- architecture diagram and parameter settings;
- training parameter settings;
- train/test loss curves;
- real online inference settings and results, both per prompt and aggregate;
- additional 30-epoch vs 100-epoch training comparison.

## Outputs

Created:

- `reports/report_gated_multifeature_mlp_predictor_comprehensive_20260630.md`
- `reports/assets/gated_multifeature_mlp_training_loss_curves.svg`

The report reuses the existing architecture diagram:

- `reports/assets/gated_multifeature_mlp_architecture.svg`

## Data Sources

Training:

- sample split:
  `/hy-tmp/wan22_adaptive_threshold_mlp_gated_5feature_range_samplesplit_20260630_035000`
- row split 30:
  `/hy-tmp/wan22_adaptive_threshold_mlp_gated_5feature_range_rowsplit_gpu_20260630_035000`
- row split 100:
  `/hy-tmp/wan22_adaptive_threshold_mlp_gated_5feature_range_rowsplit_gpu_long100_20260630_035000`

Online inference:

- 5-feature result root:
  `/hy-tmp/wan22_adaptive_seacache_mlp_gated_5feature_range_split_compare_50step_45f_480p_20260630_050727`
- MiniDiT comparison root:
  `/hy-tmp/wan22_adaptive_seacache_mini_dit_split_compare_50step_45f_480p_20260630_025328`

## Report Contents

The report includes:

- high-level architecture flow;
- per-feature definitions and input dimensions;
- module settings for feature encoders, condition encoder, gate head, and
  prediction head;
- parameter count breakdown:
  - total: `83,526`
  - feature encoders: `62,080`
  - condition encoder: `4,352`
  - gate head: `4,485`
  - prediction head: `12,609`
- shared and split-specific training settings;
- offline training metrics and gate-weight summaries;
- loss curve SVG with train/test Smooth L1 curves overlaid per run;
- 30-epoch vs 100-epoch row-split comparison;
- online inference protocol;
- online per-prompt table;
- online aggregate table;
- same-protocol MiniDiT comparison table;
- findings and artifact paths.

## Key Numbers Captured

Offline:

- sample split 30 best test MAE: `0.114357`
- row split 30 best test MAE: `0.077050`
- row split 100 best test MAE: `0.061031`
- row split 100 best epoch: `98`
- row split 100 best improves over row split 30 best by about `20.8%`

Online aggregate:

- OpenVid train sample target 22: `2.559x`, PSNR `22.858`
- OpenVid train sample target 28: `1.718x`, PSNR `27.365`
- OpenVid train row target 22: `2.523x`, PSNR `21.970`
- OpenVid train row target 28: `1.773x`, PSNR `26.095`
- VBench10 sample target 22: `2.229x`, PSNR `17.159`
- VBench10 sample target 28: `1.539x`, PSNR `25.460`
- VBench10 row target 22: `2.065x`, PSNR `20.810`
- VBench10 row target 28: `1.627x`, PSNR `24.484`

## Validation

- Confirmed the report and SVG exist.
- Reviewed the generated report sections.
- `git diff --check` passed for the generated report and SVG before updating
  `PROGRESS.md` and this session log.

No commit was made.
