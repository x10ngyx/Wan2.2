# 2026-06-27 SeaCache Official Alignment Recheck

Task: verify local Wan2.2 SeaCache behavior against official SeaCache Wan2.1 implementation from `https://github.com/jiwoogit/SeaCache/tree/main/Wan2.1`, and identify necessary Wan2.2 adaptations.

Reference:

- Official clone: `/hy-tmp/seacache_official_ref`
- Official commit: `3b1c688`
- Official files checked:
  - `/hy-tmp/seacache_official_ref/Wan2.1/seacache_generate.py`
  - `/hy-tmp/seacache_official_ref/Wan2.1/util_seacache.py`
- Local files checked:
  - `wan/timestep_cache.py`
  - `wan/modules/model.py`
  - `wan/text2video.py`
  - `generate.py`

Findings:

- Local timestep SeaCache matches official core behavior:
  - Uses first block's modulated norm input as the SeaCache metric feature.
  - Applies SEA filtering on middle decision steps using `scheduler.sigmas[step_index]`.
  - Uses accumulated relative L1 threshold gating.
  - Resets accumulated distance on threshold-crossing recompute and on forced ret/cutoff recompute windows, not on normal reuse.
  - Updates previous feature every call.
  - Reuses previous transformer-block residual, then still runs head/unpatchify.
  - Default forced windows are equivalent to official first and final denoising step per branch; `--seacache_use_ret_steps` maps to official first 5 steps and no final cutoff.

Necessary Wan2.2 adaptations:

- State keying is explicit `(model_stage, branch)` instead of official even/odd call counting. This is required because Wan2.2 has high/low model stages and this project composes SeaCache with CFG cache, so branch cannot be inferred safely from model-call parity.
- Local implementation is object-based and integrated into `WanModel.forward`; official code monkey-patches class-level model attributes and forward.
- Local code passes `scheduler.sigmas` into the cache object instead of attaching the scheduler to the model class.
- Local CLI exposes additional tunables (`power_const`, `eps`, `norm_mode`, optional `num_steps`) while defaults match the official practical Wan2.1 path (`power_exp=3.0`, `power_const=1.0`, `eps=1e-16`, `norm_mode=mean`).

Validation:

- `python -m py_compile wan/timestep_cache.py wan/modules/model.py wan/text2video.py generate.py` passed.
- `git diff --check` passed.
- CPU direct-load comparison showed local `_apply_sea_from_ab` and official `apply_sea_with_scheduler` had `filter_max_abs_diff 0.0` on a checked tensor/sigma schedule.
- CPU direct-load spy check showed forced ret/cutoff paths skip `_filter_feature`, while middle decision steps call it.

No GPU inference, PSNR, or official Wan2.1/Ali latency reproduction run was launched in this session.
