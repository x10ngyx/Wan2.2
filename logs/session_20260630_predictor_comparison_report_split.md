# Session Log: Predictor Comparison Report Split

Date: 2026-06-30

## Request

Move `9. Same-Protocol MiniDiT Comparison` out of the 5-feature gated MLP
comprehensive report into a dedicated comparison report. The dedicated report
should include both methods' training loss data/figures and online inference
performance comparison.

## Changes

Updated:

- `reports/report_gated_multifeature_mlp_predictor_comprehensive_20260630.md`
  - removed the old same-protocol MiniDiT comparison section.
  - renumbered `Findings` and `Artifacts`.
  - kept the report focused on 5-feature architecture, training, curves, and
    online results.
  - artifact table now references the dedicated comparison report.

Created:

- `reports/report_adaptive_predictor_mini_dit_vs_gated_mlp_comparison_20260630.md`

## Dedicated Comparison Report Contents

The new comparison report includes:

- method summary:
  - MiniDiT-CLS Transformer: `724,513` params.
  - 5-feature gated MLP: `83,526` params.
- training loss data table:
  - MiniDiT sample split 30.
  - MiniDiT row split 30.
  - 5-feature MLP sample split 30.
  - 5-feature MLP row split 30.
  - 5-feature MLP row split 100.
- online-checkpoint training comparison:
  - MiniDiT sample checkpoint: test MAE `0.114459`.
  - 5-feature sample checkpoint: test MAE `0.114357`.
  - MiniDiT row checkpoint: test MAE `0.038002`.
  - 5-feature row checkpoint: test MAE `0.061031`.
- training loss figures:
  - `reports/assets/mini_dit_cls_training_loss_curves.svg`
  - `reports/assets/gated_multifeature_mlp_training_loss_curves.svg`
- shared online inference protocol.
- online aggregate comparison table.
- online per-prompt comparison table.
- findings and artifact paths.

## Validation

- Reviewed the updated 5-feature report section numbering.
- Reviewed the new comparison report tables and figure references.
- Corrected MiniDiT sample split display from missing config field `None` to
  `sample`.
- No new experiments were run.

No commit was made.
