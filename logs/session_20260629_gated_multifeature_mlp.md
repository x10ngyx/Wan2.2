# 2026-06-29 Gated Multi-Feature MLP

## Summary

- Implemented Scheme B: Per-Feature MLP + Gated Fusion for the legacy MLP adaptive threshold predictor.
- Preserved the previous direct-concat multi-feature path as `--feature_fusion concat`.
- Added `--feature_fusion gated`, defaulting to the four requested features:
  - `latent_pool`
  - `temporal_var`
  - `frame_diff_mean`
  - `frame_diff_var`

## Files Changed

- `adaptive_threshold_predictor/models.py`
  - Added `DEFAULT_GATED_FEATURE_SETS`.
  - Added `GatedFeatureFusionAdaCacheGate`.
  - Added `CachedGatedFeatureAdaCacheGate`.
  - Added `GatedMultiFeatureAdaCacheGate`.
- `adaptive_threshold_predictor/data.py`
  - `CachedFeatureThresholdDataset` now keeps per-feature tensors in addition to the direct concatenated tensor.
  - `collate_cached_features` now returns `batch["features"][feature_name]`.
- `adaptive_threshold_predictor/train_gate.py`
  - Added `--feature_fusion concat|gated`.
  - Added `--feature_embedding_dim`.
  - Routed cached and raw-latent gated MLP runs through the new gated models.
  - Added resolved feature metadata to config/metrics.
  - Added gated columns to `val_predictions.csv` for gated models.
- `adaptive_threshold_predictor/README.md`
  - Added the recommended gated 4-feature MLP command.
- `PROGRESS.md`
  - Added this session's implementation and validation summary.

## Validation

- Syntax:
  - `/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m py_compile adaptive_threshold_predictor/data.py adaptive_threshold_predictor/models.py adaptive_threshold_predictor/train_gate.py`
- Raw-latent forward smoke:
  - `GatedMultiFeatureAdaCacheGate(latent_channels=16, hidden_dim=16, feature_embedding_dim=8)`
  - random input `[2, 16, 12, 60, 104]`
  - output shape `[2, 1]`
  - gate shape `[2, 4]`
  - gate rows summed to 1.
- Cached gated CPU smoke:
  - cache: `/hy-tmp/wan22_adaptive_threshold_feature_cache_candidate_inverse_20260616_012409`
  - command used `--feature_fusion gated`
  - `max_examples=256`, `epochs=1`, `device=cpu`
  - output: `/tmp/wan22_mlp_gated_4feature_smoke`
- Cached gated CPU smoke with validation predictions:
  - `max_examples=1200`, `epochs=1`, `device=cpu`, `--save_val_predictions`
  - output: `/tmp/wan22_mlp_gated_4feature_val_smoke`
  - `val_predictions.csv` includes `gate_latent_pool`, `gate_temporal_var`, `gate_frame_diff_mean`, `gate_frame_diff_var`.
- `git diff --check` passed.

## Notes

- No full GPU training run was launched.
- No git commit was made.
