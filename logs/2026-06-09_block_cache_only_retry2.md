# 2026-06-09 Block-cache-only retry2

- The previous retry session `block_cache_only_p01_retry_1318` was started before commit `4ad3615`, so it was stopped to ensure the run uses early stage clearing and summary archiving.
- Archived its partial `bwcache_th_0p05` log/time under `/hy-tmp/wan22_block_cache_only_50step_45f_480p_20260609_125436/failed_history/`.
- Restarted tmux session `block_cache_only_p01_retry2_1330` with the same experiment root and `RESUME_EXISTING=True`.
- Initial pane confirmed baseline was skipped and `bwcache_th_0p05` restarted.
