# 2026-06-15 TaylorSeer Timestep Prototype

## Summary

- Implemented a lightweight TaylorSeer-style timestep-output cache for WanT2V.
- Added `--timestep_cache taylorseer` with:
  - `--taylorseer_interval`
  - `--taylorseer_order`
  - `--taylorseer_ret_steps`
  - `--taylorseer_cutoff_steps`
- Integrated the cache through the existing T2V branch-forward path with explicit `(model_stage, branch)` keys.

## Files Changed

- `wan/timestep_cache.py`
  - Added `TaylorSeerTimestepCacheConfig`, `TaylorSeerTimestepCacheState`, and `TaylorSeerTimestepCache`.
  - Uses recent recomputed denoiser outputs and polynomial/Taylor-style extrapolation for skipped timestep outputs.
- `wan/text2video.py`
  - Instantiates and dispatches TaylorSeer timestep cache when selected.
- `generate.py`
  - Exposes CLI args and builds the TaylorSeer config.
- `PROGRESS.md`
  - Appended implementation and validation notes.

## Validation

- Passed syntax compilation in the project conda env:
  - `/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m py_compile generate.py wan/timestep_cache.py wan/text2video.py`
- Passed file-level cache behavior check without importing the full `wan` package:
  - `interval=3`, `order=1`, `ret_steps=1`, `cutoff_steps=1`
  - recompute steps: `[0, 1, 4, 7]`
  - reuse steps: `[2, 3, 5, 6]`

## Notes

- Current machine boot has no visible GPU; `nvidia-smi` returned `No devices were found`.
- No real Wan2.2 generation was launched.
- This is a timestep-output-level Taylor forecasting baseline, not a full official TaylorSeer hidden-state/block-level reproduction.
