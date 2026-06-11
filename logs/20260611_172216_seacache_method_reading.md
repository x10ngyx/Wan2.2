# 2026-06-11 SeaCache method reading

- User asked for a method-only explanation of the timestep cache implementation in `https://jiwoogit.github.io/SeaCache/`.
- Read the project page and cloned `https://github.com/jiwoogit/SeaCache.git` to `/tmp/seacache_read` for local inspection.
- Focused on `Wan2.1/seacache_generate.py` and `Wan2.1/util_seacache.py`, with quick cross-checks of HunyuanVideo and FLUX variants.
- Main understanding: SeaCache decides whole-transformer timestep reuse from scheduler-aware frequency-filtered, time-modulated input features; it accumulates relative-L1 changes since the last recompute and reuses the previously cached transformer residual while the accumulated change remains under the threshold.
- No experiment code was modified and no GPU jobs were launched.
