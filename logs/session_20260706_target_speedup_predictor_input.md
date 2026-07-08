# Session Log: Target Speedup Predictor Input

Date: 2026-07-06

## Summary

- Added `target_speedup` as a predictor condition input, handled like `target_psnr`, for both the 5-feature gated MLP path and MiniDiT/Transformer path.
- Updated predictor training, evaluation, prediction CSV output, cache builders, inspection utility, README, and online adaptive SeaCache wiring.
- Removed `target_oracle` / `dataset_mode` from the core training and cache-building pipeline.
- Confirmed the OpenVid-100 SeaCache trace summary records finite speedup for all `1000/1000` candidate rows.

## Implementation Notes

- Predictor condition vector is now `(timestep_norm, target_psnr_norm, target_speedup_norm)`.
- Speedup normalization uses `clamp((target_speedup - speedup_min) / (speedup_max - speedup_min), 0, 1)` with defaults `speedup_min=1.0` and `speedup_max=4.0`.
- New cache metadata writes `target_speedup`; old raw/feature/grid caches can recover speedup through `source_index` against the summary rows.
- Online adaptive SeaCache now requires `--target_speedup`; historical runner configs were given a compatibility default of `2.0`.
- Existing two-condition checkpoints are not shape-compatible with the new three-condition models and should be retrained.

## Files Touched

- `adaptive_threshold_predictor/data.py`
- `adaptive_threshold_predictor/models.py`
- `adaptive_threshold_predictor/train_gate.py`
- `adaptive_threshold_predictor/build_feature_cache.py`
- `adaptive_threshold_predictor/build_raw_latent_cache.py`
- `adaptive_threshold_predictor/build_grid_feature_cache.py`
- `adaptive_threshold_predictor/run_feature_ablation.py`
- `adaptive_threshold_predictor/run_grid_ablation.py`
- `adaptive_threshold_predictor/inspect_trace_data.py`
- `adaptive_threshold_predictor/README.md`
- `adaptive_seacache_wan22/cache.py`
- `adaptive_seacache_wan22/generate_t2v.py`
- Historical adaptive SeaCache runner scripts under `experiments/adaptive_seacache_*`.
- `PROGRESS.md`

## Verification

- `python -m compileall adaptive_threshold_predictor adaptive_seacache_wan22 experiments/adaptive_seacache_mini_dit_split_compare_50step_45f_480p/run_batch.py`
- `python -m adaptive_threshold_predictor.inspect_trace_data --target_psnr 28 --target_speedup 2.0`
- Raw dataset and old cached-feature dataset smoke check confirmed `target_speedup` is present and batched correctly.
- CPU smoke training completed for 5-feature gated MLP using the existing feature cache.
- CPU smoke training completed for MiniDiT using raw examples.
- Residual search found no remaining `DATASET_MODES`, `target_oracle`, `--dataset_mode`, or `dataset_mode` in the relevant predictor/adaptive paths.

## Follow-Up

- Retrain the intended sample-split/row-split predictors with the new target speedup condition.
- Update downstream experiment launch configs to choose explicit target speedup values instead of relying on compatibility defaults in historical scripts.
