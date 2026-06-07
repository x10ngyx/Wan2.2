# 2026-06-08 ZEUS Threshold Implementation

- User requested a new `zeus-threshold` timestep cache while keeping the original official ZEUS implementation unchanged.
- Added new classes in `wan/timestep_cache.py`:
  - `ZeusThresholdTimestepCacheConfig`
  - `ZeusThresholdTimestepCacheState`
  - `ZeusThresholdTimestepCache`
- Original `ZeusTimestepCacheConfig`, `ZeusTimestepCacheState`, and `ZeusTimestepCache` definitions were left unchanged.
- `zeus-threshold` keeps the existing state keying by `(model_stage, branch)` and existing output reconstruction modes.
- The fixed ZEUS skip schedule is replaced only in the new threshold path: current input latent relative-L1 distance is compared directly with `threshold`; no accumulated distance is used.
- Added `--timestep_cache zeus-threshold` and threshold CLI parameters in `generate.py`.
- Wired `WanT2V.generate()` to pass `latent_model_input[0]` only to the threshold cache path; the original `zeus` call shape remains unchanged.
- Validation:
  - `/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m py_compile generate.py wan/text2video.py wan/timestep_cache.py`
  - `python -m py_compile generate.py wan/text2video.py wan/timestep_cache.py`
  - CPU dummy cache checks using import-by-path to avoid CUDA import-time initialization.
