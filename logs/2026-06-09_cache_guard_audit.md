# 2026-06-09 cache guard audit

- Audited first/last denoising protection and maximum consecutive reuse controls across the active cache implementations.
- Existing guards:
  - timestep cache: `acc_range` and `max_interval`
  - CFG cache: `cfg_start`, `cfg_end`, and `cfg_max_reuse`
  - block-group cache: `block_group_start`, `block_group_end`, and `block_group_max_reuse`
  - BWCache: tail protection through `last_step` and max consecutive reuse through `reuse_interval`
- Gap found: BWCache did not have an explicit start/end progress window.
- Added BWCache `start` / `end` config fields and CLI flags:
  - `--bwcache_start`
  - `--bwcache_end`
- BWCache schedule generation now forces recompute outside the configured progress window while preserving the existing reuse-interval pattern and tail protection.
- Defaults are `0.0` and `1.0`, so existing BWCache commands keep their old behavior unless a window is explicitly set.
- Validation:
  - `/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m py_compile generate.py wan/block_cache.py wan/block_group_cache.py wan/cfg_cache.py wan/timestep_cache.py wan/text2video.py`
  - `python -m py_compile generate.py wan/block_cache.py wan/block_group_cache.py wan/cfg_cache.py wan/timestep_cache.py wan/text2video.py`
  - `/hy-tmp/miniconda3/envs/Wan2.2/bin/python generate.py --help`
  - CPU BWCache schedule check for protected prefix/suffix and max consecutive reuse pattern
- No GPU generation or PSNR experiment was launched in this session.
