# 2026-06-09 CFG miss force uncond recompute switch

- User asked for a switch so CFG miss can force recomputation and update timestep/block cache, addressing stale uncond cache state after CFG skips.
- Added `force_recompute` support to `ZeusTimestepCache.call()` and `ZeusThresholdTimestepCache.call()`. Forced calls bypass skip decisions but still call `record_recompute()`, updating cache state.
- Added `block_cache_force_recompute` to `WanModel.forward()`. Forced calls bypass BWCache residual reuse but still run all blocks, update `previous_residual`, append recompute stats, and run the BWCache schedule update path.
- Added `CFGCacheConfig.force_uncond_recompute_on_miss` and CLI flag `--cfg_force_uncond_recompute_on_miss`. In `WanT2V.generate()`, CFG miss passes this force flag only to the `uncond` branch. Default remains off.
- Validation: conda py_compile passed for changed files; generate.py --help showed the new flag; CPU ZEUS cache check verified normal skip and forced recompute behavior. No full GPU inference/PSNR run was launched.
