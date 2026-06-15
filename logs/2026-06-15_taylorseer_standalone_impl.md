# 2026-06-15 TaylorSeer Standalone Implementation

## Summary

- Reverted TaylorSeer changes from main Wan2.2 code paths:
  - `generate.py`
  - `wan/text2video.py`
  - `wan/timestep_cache.py`
  - `wan/modules/model.py`
- Added a standalone TaylorSeer implementation under `taylorseer_wan22/`.
- Main ZEUS, SeaCache, block-cache, and CFG-cache composition logic is not touched by the standalone implementation.

## Standalone Files

- `taylorseer_wan22/cache.py`
  - Official-style scheduler/cache state:
    - `fresh_threshold`
    - `max_order`
    - `first_enhance`
    - `cache_counter`
    - `activated_steps`
  - Separate cond/uncond stream caches.
- `taylorseer_wan22/patch.py`
  - Runtime monkey patch for transformer block forward.
  - Caches/predicts self-attention, cross-attention, and FFN module outputs.
- `taylorseer_wan22/text2video.py`
  - Subclasses existing `WanT2V`.
  - Installs TaylorSeer patches only in the standalone pipeline.
  - Keeps high and low denoiser stages as separate TaylorSeer cache states.
- `taylorseer_wan22/generate_t2v.py`
  - Standalone T2V runner for TaylorSeer experiments.

## Low/High Behavior

- Wan2.2 uses separate `high_noise_model` and `low_noise_model`.
- The standalone TaylorSeer implementation installs patches on both models but uses separate cache keys:
  - `high`
  - `low`
- No module cache is shared across the high/low model boundary.
- Within a stage, cond decides full/Taylor scheduling first, matching official Wan2.1 behavior; uncond follows the same step type while keeping a separate stream cache.

## Validation

- Passed:
  - `/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m py_compile taylorseer_wan22/cache.py taylorseer_wan22/patch.py taylorseer_wan22/text2video.py taylorseer_wan22/generate_t2v.py generate.py wan/text2video.py wan/timestep_cache.py wan/modules/model.py`
- Confirmed no TaylorSeer strings remain in main code paths:
  - `rg -n "TaylorSeer|taylorseer" generate.py wan/text2video.py wan/timestep_cache.py wan/modules/model.py`
- No real generation was launched because the current boot has no visible GPU.
