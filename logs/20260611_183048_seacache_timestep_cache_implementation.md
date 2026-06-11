# 2026-06-11 SeaCache Timestep Cache Implementation

- User requested complete reproduction of SeaCache timestep cache from `jiwoogit/SeaCache`, added as a new timestep cache method without affecting existing methods, with Wan2.2 high/low switching handled by cold start.
- Re-read the public SeaCache page and local clone `/tmp/seacache_read`, focusing on `Wan2.1/seacache_generate.py` and `Wan2.1/util_seacache.py`.
- Implemented `SeaCacheTimestepCacheConfig`, `SeaCacheTimestepCacheState`, and `SeaCacheTimestepCache` in `wan/timestep_cache.py`.
- Integrated SeaCache into `WanModel.forward()` as transformer-block residual reuse after patch/time/context embedding and before head/unpatchify, matching the original Wan2.1 SeaCache behavior instead of reusing final denoiser output.
- Wired `WanT2V.generate()` so `--timestep_cache seacache` uses explicit `(model_stage, branch)` keys. This gives independent high/low and cond/uncond states, so the Wan2.2 high-to-low switch cold-starts naturally.
- Added `generate.py` CLI support for `--timestep_cache seacache` plus `--seacache_threshold`, `--seacache_num_steps`, `--seacache_use_ret_steps`, `--seacache_power_exp`, `--seacache_power_const`, `--seacache_eps`, and `--seacache_norm_mode`.
- Validation passed: conda `py_compile` for `wan/timestep_cache.py`, `wan/modules/model.py`, `wan/text2video.py`, and `generate.py`; CPU by-path state-machine test confirmed same-key reuse and low-stage cold start.
- Validation limitation: current shell has no visible GPU (`nvidia-smi` reports `No devices were found`), so `generate.py --help` and a real T2V smoke were not run because normal `wan` import reaches `torch.cuda.current_device()`.
