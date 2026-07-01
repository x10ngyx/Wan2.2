# 2026-06-29 Gated MLP No-Concat Revision And Report

## Summary

- Removed the multi-feature direct-concat path.
- Kept single-feature legacy MLP behavior through `--feature_set`.
- Multi-feature MLP behavior is now gated fusion only, selected by passing more than one `--feature_sets` value.
- Calculated the recommended 4-feature gated MLP parameter count.
- Added an architecture report modeled after the Transformer predictor report.

## Files Changed

- `adaptive_threshold_predictor/models.py`
  - Restored `ImprovedAdaCacheGate` to single-feature behavior only.
  - Kept gated multi-feature classes as the only multi-feature MLP implementation.
- `adaptive_threshold_predictor/data.py`
  - Stopped constructing a concatenated multi-feature tensor.
  - Multi-feature cached datasets now return per-feature tensors only.
- `adaptive_threshold_predictor/train_gate.py`
  - Removed `--feature_fusion concat|gated`.
  - `--feature_sets` with more than one feature now routes to gated fusion.
  - `--feature_set` remains the single-feature legacy path.
- `adaptive_threshold_predictor/README.md`
  - Removed the concat multi-feature command.
  - Documented gated fusion as the multi-feature MLP path.
- `reports/report_gated_multifeature_mlp_architecture.md`
  - Added architecture, hyperparameters, parameter count, training command, and diagnostics.
- `PROGRESS.md`
  - Added this revision summary.

## Parameter Count

Recommended first-run settings:

- features: `latent_pool temporal_var frame_diff_mean frame_diff_var`
- per-feature input dim: `128`
- hidden dim: `64`
- feature embedding dim: `64`

Measured trainable parameters:

- total: `71,045`
- feature encoders total: `49,664`
- each feature encoder: `12,416`
- condition encoder: `4,352`
- gate head: `4,420`
- prediction head: `12,609`

## Validation

- Syntax:
  - `/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m py_compile adaptive_threshold_predictor/data.py adaptive_threshold_predictor/models.py adaptive_threshold_predictor/train_gate.py`
- Cached gated CPU smoke:
  - `--feature_sets latent_pool temporal_var frame_diff_mean frame_diff_var`
  - `max_examples=64`, `epochs=1`, `device=cpu`
  - output: `/tmp/wan22_mlp_gated_no_concat_smoke`
- Cached gated CPU smoke with validation predictions:
  - `max_examples=1200`, `epochs=1`, `device=cpu`, `--save_val_predictions`
  - output: `/tmp/wan22_mlp_gated_no_concat_val_smoke`
  - prediction CSV includes `gate_latent_pool`, `gate_temporal_var`, `gate_frame_diff_mean`, and `gate_frame_diff_var`.

## Notes

- No full GPU training was launched.
- No git commit was made.
