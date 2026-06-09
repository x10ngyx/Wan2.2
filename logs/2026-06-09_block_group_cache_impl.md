# 2026-06-09 Block-group cache implementation

- User requested a new block cache option, separate from the existing BWCache implementation, using 5 transformer blocks per group across 8 groups for T2V-A14B.
- Added `wan/block_group_cache.py` with `BlockGroupCacheConfig`, per-group state, threshold-based relative-L1 reuse checks, pooled/full metric options, max reuse streaks, and summary output.
- Added `WanAttentionBlock.block_group_cache_feature()` and a separate block-group branch in `WanModel.forward()`. The new path groups blocks by `group_size`, reuses cached group residuals on threshold hits, recomputes the full group on misses, and does not alter the existing BWCache branch.
- Wired `WanT2V.generate()` and `generate.py` for `--block_cache block-group` plus group-specific CLI parameters. CFG miss force recompute now also forces block-group recompute for the uncond branch.
- Validation: conda py_compile passed for changed cache/model/generate files; `generate.py --help` shows the new block-group options; a CPU BlockGroupCache logic check verified threshold hit/miss and summary fields. No full GPU inference or PSNR run was launched.
