# 2026-06-16 Adaptive Predictor Top-level Move

## Summary

- Deleted `framework.png`.
- Moved adaptive predictor code out of `experiments/` into top-level `adaptive_threshold_predictor/`.
- Updated imports and module invocation strings to use `adaptive_threshold_predictor.*`.
- This reflects that adaptive prediction is now a main project direction rather than a one-off experiment.

## Files / Directories Changed

- Removed: `framework.png`
- Moved: `experiments/adaptive_threshold_predictor/` -> `adaptive_threshold_predictor/`
- Updated: `adaptive_threshold_predictor/README.md`
- Updated: `PROGRESS.md`
- Added: `logs/2026-06-16_adaptive_predictor_top_level_move.md`

## Validation

- Validation commands were run after the move and are reported in the final assistant response.
