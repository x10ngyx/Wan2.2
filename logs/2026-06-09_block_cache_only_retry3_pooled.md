# 2026-06-09 block-cache-only retry3 with pooled BWCache

- Rechecked the active experiment after the BWCache pooled-feature change.
- Active tmux session: `block_cache_only_p01_retry3_1340`.
- Result root: `/hy-tmp/wan22_block_cache_only_50step_45f_480p_20260609_125436`.
- Archived the previous in-flight pre-pooling attempt as `failed_history/bwcache_th_0p05_prompt_01.interrupted_before_bwcache_pool_20260609_133049.log`.
- `experiment.env` and `thresholds/bwcache_th_0p05.env` confirm `bwcache_metric=pooled_rel_l1`.
- Pane confirmed baseline prompt 01 was skipped and `bwcache_th_0p05` restarted with `--block_cache bwcache`, `--bwcache_thresh 0.05`, timestep cache disabled, and CFG cache disabled.
