# 2026-06-09 Block-cache OOM fix

- The block-cache-only tmux run exited during `bwcache_th_0p05` with CUDA OOM at sampling step 32.
- Log: `/hy-tmp/wan22_block_cache_only_50step_45f_480p_20260609_125436/logs/bwcache_th_0p05_prompt_01.log`.
- Failure record: `/hy-tmp/wan22_block_cache_only_50step_45f_480p_20260609_125436/failed/bwcache_th_0p05_prompt_01.txt`.
- Root cause: BWCache retained high-stage cond/uncond block feature/residual state after denoising switched to low stage. Since Wan timesteps are monotonic from high to low, completed high-stage block cache state is dead and safe to clear.
- Added `clear_stage(stage)` to `wan/block_cache.py` and `wan/block_group_cache.py`.
- Wired `WanT2V.generate()` to clear completed block-cache stage state when `model_stage` changes.
- Validation: conda `py_compile` passed for touched files, and a CPU state check verified high-stage entries are removed while low-stage entries remain.
