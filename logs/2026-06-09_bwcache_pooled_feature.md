# 2026-06-09 BWCache pooled feature update

- User noted BWCache should pool cached block features, aligned with the block-group implementation.
- Added `metric` to `BWBlockCacheConfig`, defaulting to `pooled_rel_l1`; `full_rel_l1` remains available.
- Updated `WanAttentionBlock._record_bwcache_l1()` so pooled mode stores `x_m.float().mean(dim=1)` as the previous block feature instead of the full token feature.
- Added CLI `--bwcache_metric` and updated block-cache-only experiment scripts/README to pass and record `BWCACHE_METRIC=pooled_rel_l1`.
- Validation: conda py_compile passed, `generate.py --help` shows the new flag, and `bash -n` passed for the experiment runner.
