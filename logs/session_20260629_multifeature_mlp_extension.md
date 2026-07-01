# 2026-06-29 Multi-Feature MLP Extension

## Summary

- Read the `adaptive_threshold_predictor/` package to understand the legacy MLP input path.
- Extended the legacy MLP path from one selectable latent-derived feature to multiple concatenated features.
- Preserved `--feature_set` for existing single-feature runs.
- Added `--feature_sets` for multi-feature runs.

## Files Changed

- `adaptive_threshold_predictor/models.py`
  - Added `normalize_feature_sets`.
  - Updated `ImprovedAdaCacheGate` to accept `feature_sets`, extract each feature, concatenate pooled feature vectors, and keep the prediction head unchanged.
- `adaptive_threshold_predictor/data.py`
  - Updated `CachedFeatureThresholdDataset` to load and concatenate multiple `features_<feature>.pt` files from a feature cache.
- `adaptive_threshold_predictor/train_gate.py`
  - Added `--feature_sets`.
  - Wired selected feature lists into cached and raw-latent MLP paths.
  - Recorded `feature_sets` in metrics and feature-extractor config.
- `adaptive_threshold_predictor/README.md`
  - Added Multi-Feature MLP usage notes and an example command.
- `PROGRESS.md`
  - Added this session's implementation and validation summary.

## Validation

- Syntax:
  - `/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m py_compile adaptive_threshold_predictor/data.py adaptive_threshold_predictor/models.py adaptive_threshold_predictor/train_gate.py`
- Cached single-feature smoke:
  - `--feature_set latent_pool`
  - `--cache_dir /hy-tmp/wan22_adaptive_threshold_feature_cache_candidate_inverse_20260616_012409`
  - `--max_examples 256 --epochs 1 --device cpu`
  - output: `/tmp/wan22_mlp_single_feature_smoke`
- Cached multi-feature smoke:
  - `--feature_sets latent_pool temporal_var frame_diff_mean`
  - same cache and smoke settings
  - output: `/tmp/wan22_mlp_multi_feature_smoke`
- Raw-latent forward smoke:
  - instantiated `ImprovedAdaCacheGate(feature_sets=("latent_pool", "temporal_var", "frame_diff_mean"))`
  - random input shape `[2, 16, 12, 60, 104]`
  - output shape `[2, 1]`

## Notes

- The CPU smoke runs used only the first 256 examples, which all belong to a small subset of samples, so validation buckets were empty. This was intentional for fast path validation.
- No GPU training or full multi-feature ablation was launched.
- No git commit was made.
