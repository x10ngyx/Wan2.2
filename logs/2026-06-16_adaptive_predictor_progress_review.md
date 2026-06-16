# 2026-06-16 Adaptive Predictor Progress Review

## Summary

- Read the adaptive-threshold predictor logs, repository files, and main result summaries.
- Focused on current progress, dataset construction, model/training code, completed ablations, and remaining gaps.
- Did not change predictor code or rerun training.

## Files Reviewed

- `PROGRESS.md`
- `adaptive_threshold_predictor/README.md`
- `adaptive_threshold_predictor/data.py`
- `adaptive_threshold_predictor/models.py`
- `adaptive_threshold_predictor/train_gate.py`
- `adaptive_threshold_predictor/build_feature_cache.py`
- `adaptive_threshold_predictor/run_feature_ablation.py`
- `adaptive_threshold_predictor/run_grid_ablation.py`
- `adaptive_threshold_predictor/inspect_trace_data.py`
- `logs/2026-06-15_adaptive_threshold_predictor_scaffold.md`
- `logs/2026-06-16_adaptive_control_baselines.md`
- `logs/2026-06-16_adaptive_pooling_grid_ablation.md`

## Result Roots Checked

- `/hy-tmp/wan22_adaptive_threshold_feature_cache_candidate_inverse_20260616_012409`
- `/hy-tmp/wan22_adaptive_threshold_feature_ablation_cached_20260616_012409`
- `/hy-tmp/wan22_adaptive_threshold_controls_20260616`
- `/hy-tmp/wan22_adaptive_threshold_grid_ablation_20260616_020314`

## Current State

- Predictor work is separate from Wan generation runners under `adaptive_threshold_predictor/`.
- Current scope is timestep/SeaCache-threshold prediction only.
- It does not yet predict the full `(timestep_threshold, block_threshold, cfg_threshold)` three-cache combination.
- Default dataset mode is `candidate_inverse`.
- Current input row is candidate latent at one denoising step, normalized step index, and achieved candidate PSNR.
- Current label is the SeaCache threshold used by that candidate run.
- Dataset size is `100 samples * 10 thresholds * 50 steps = 50000` examples.
- Train/validation split is grouped by `sample_id`, producing `40000` train examples and `10000` validation examples with no sample overlap.

## Current Best Result

- Current loss-based default remains `2x2x2 temporal_mean`.
- Output root: `/hy-tmp/wan22_adaptive_threshold_feature_ablation_cached_20260616_012409`
- Best validation loss: `0.012259`
- Validation MAE: `0.120107`
- Parameters: `29377`

## Reviewed Comparisons

- `2x2x2 latent_pool` has slightly lower MAE (`0.116558`) but worse best val loss (`0.012755`) than `2x2x2 temporal_mean`.
- Larger grid ablation did not beat `2x2x2 temporal_mean` by validation loss.
- Best larger-grid result was `2x4x4 latent_pool` with best val loss `0.012434`, MAE `0.118093`, params `53953`.
- Control baselines show timestep plus PSNR explain much of the label structure:
  - `condition_only`: best val loss `0.014652`, MAE `0.128916`.
  - `noise_feature`: best val loss `0.014648`, MAE `0.131173`.
- Real features still improve validation loss by about `13%` to `16%` relative to controls.

## Gaps

- Training runs are short 3-epoch, single-split early-stopping ablations, not converged multi-seed results.
- The current model predicts direct threshold labels, but there is not yet an adaptive inference runner that consumes predictions to generate videos and verify achieved PSNR/speedup.
- Current work is SeaCache/timestep-threshold-only and has not yet been extended to the project goal of predicting three-cache threshold combinations.
- No target-quality/target-speed joint interface is implemented yet; current conditioning uses achieved or desired PSNR depending on dataset mode, with no speed target input.

## Files Changed

- Appended a review note to `PROGRESS.md`.
- Added this session log.
