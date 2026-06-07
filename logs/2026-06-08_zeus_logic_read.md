# 2026-06-08 ZEUS Logic Read

- User asked to carefully read the repository's current ZEUS implementation logic.
- Read `PROGRESS.md` first per project workflow.
- Reviewed `wan/timestep_cache.py`, `wan/text2video.py`, `generate.py`, and `experiments/zeus_timestep_cache_50step_45f_480p/`.
- Current implementation is ZEUS-style timestep cache for native Wan2.2 T2V only.
- `generate.py` exposes `--timestep_cache {none,zeus}` and ZEUS scheduling/reuse parameters, then passes `ZeusTimestepCacheConfig` into `WanT2V.generate()`.
- `WanT2V.generate()` wraps the existing cond and uncond model calls with `ZeusTimestepCache.call(...)` when enabled.
- Cache state keys are `(model_stage, branch)`, separating `high/low` model stages and `cond/uncond` CFG branches.
- `wan/timestep_cache.py` requires three recomputed outputs before skip eligibility, applies acc-range/modular/lagrange/max-interval gates, and reconstructs skipped model outputs by extrapolation or last-output reuse depending on `caching_mode`.
- `--block_cache none` and `--cfg_cache none` are placeholders only; no block-cache or CFG-cache logic is currently active.
