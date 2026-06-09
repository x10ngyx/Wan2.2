# 2026-06-09 Block-cache-only retry

- After adding and committing stage-switch block-cache clearing (`689f67b`), archived the failed `bwcache_th_0p05` attempt under `/hy-tmp/wan22_block_cache_only_50step_45f_480p_20260609_125436/failed_history/`.
- Removed the active failed marker/time/log so `RESUME_EXISTING=True` would rerun `bwcache_th_0p05` instead of leaving stale failure state.
- Restarted tmux session `block_cache_only_p01_retry_1318` with the same experiment root.
- Initial pane confirmed baseline was skipped and `bwcache_th_0p05` restarted.
