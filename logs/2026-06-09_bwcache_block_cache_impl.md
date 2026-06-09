# 2026-06-09 BWCache block cache implementation

- User requested initial block cache implementation following the official `hsc113/BWCache` repository and respecting AGENTS cache composition logic.
- Cloned official BWCache to `/tmp/BWCache` for inspection. Key official semantics used: manager config `thresh/reuse_interval/last_step`; per forward `cal_list`; reuse by `x += previous_residual`; recompute by running all transformer blocks, accumulating per-block relative-L1 of modulated attention input, then generating a future `[0] * reuse_interval + [1]` recompute pattern and tail recompute mask.
- Added `wan/block_cache.py` with `BWBlockCacheConfig`, per-key state, schedule update, and summary output.
- Modified `wan/modules/model.py` so `WanAttentionBlock` can optionally record official-style L1 and `WanModel.forward` can skip/recompute the transformer block stack with BWCache residual reuse.
- Modified `wan/text2video.py` so block cache is constructed per generation and passed only through actual cond/uncond model forwards. Timestep cache still wraps whole model calls, so timestep hits bypass BWCache; states are keyed by explicit `(model_stage, branch)`.
- Modified `generate.py` to add `--block_cache bwcache`, `--bwcache_thresh`, `--bwcache_reuse_interval`, and `--bwcache_last_step`.
- Validation: conda `py_compile` passed for changed files; `generate.py --help` passed and showed the new CLI; small CPU schedule check produced expected official pattern. No full inference or PSNR run was launched.
