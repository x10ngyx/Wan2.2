# 2026-06-13 Sea-Style CFG Cache Implementation

## Scope

Implemented a new cfg-cache-only method aligned with SeaCache:

- SEA-style scheduler-aware frequency filtering.
- Accumulated relative-L1 threshold.
- First-step and tail-step recompute protection only.

## Files Changed

- `wan/cfg_cache.py`
  - Added `SeaCFGCacheConfig`.
  - Added `SeaCFGCache`.
  - Reused CFG delta reconstruction semantics from the existing CFG cache.
  - Added accumulated distance paths to cache summaries.
- `wan/text2video.py`
  - Instantiates `SeaCFGCache` when `cfg_cache_config` is `SeaCFGCacheConfig`.
  - Uses `model.seacache_feature(...)` to obtain the first-block modulated norm feature and grid size.
  - Passes scheduler sigmas into the Sea CFG cache.
- `generate.py`
  - Added `--cfg_cache sea-threshold`.
  - Added `--cfg_sea_power_exp`, `--cfg_sea_power_const`, `--cfg_sea_norm_mode`, `--cfg_ret_steps`, and `--cfg_cutoff_steps`.
  - Clarified that `--cfg_max_reuse` applies to the old threshold CFG cache, not `sea-threshold`.
- `PROGRESS.md`
  - Documented the new method and validation.

## Behavior

`--cfg_cache sea-threshold` skips the uncond branch when accumulated SEA-filtered relative-L1 remains below `--cfg_threshold`. It forces recompute for the first `--cfg_ret_steps` global denoising steps and final `--cfg_cutoff_steps` global denoising steps. It does not apply `--cfg_start/--cfg_end` or a fixed reuse cadence.

## Validation

Ran:

```bash
/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m py_compile wan/cfg_cache.py wan/text2video.py generate.py
/hy-tmp/miniconda3/envs/Wan2.2/bin/python generate.py --help | rg -n "sea-threshold|cfg_sea|cfg_ret|cfg_cutoff"
```

Also ran a small CPU state-machine check for `SeaCFGCache` confirming:

- first-step protection,
- tail-step protection,
- accumulated distance tracking,
- consecutive reuse is not limited by `cfg_max_reuse`.

No GPU inference was launched.

## Next

Run a cfg-cache-only sweep with `--timestep_cache none --block_cache none --cfg_cache sea-threshold` against prompt-01/prompt-02 before launching an OpenVid shard.
