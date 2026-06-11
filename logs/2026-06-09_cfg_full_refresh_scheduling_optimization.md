# 2026-06-09 CFG full-refresh scheduling optimization

- Followed up on the internal-metric rerun where speedup dropped to `1.061x`.
- Issue: with `timestep_modulated_input_rel_l1`, CFG reuse no longer depends on cond output, but the implementation still computed `cond` first. On CFG miss with full refresh enabled, it then recomputed `cond`, causing duplicate cond work on miss steps.
- Reordered the non-`cond_output_rel_l1` CFG path:
  - compute model-internal CFG metric feature first
  - decide CFG hit/miss from that feature
  - on hit: compute `cond` once and skip `uncond`
  - on miss: compute forced `cond` once and forced `uncond` once, then refresh CFG delta
- Kept the `cond_output_rel_l1` path unchanged because that metric requires cond output before the CFG decision.
- Validation:
  - `/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m py_compile generate.py wan/text2video.py wan/cfg_cache.py wan/modules/model.py`
  - `python -m py_compile generate.py wan/text2video.py wan/cfg_cache.py wan/modules/model.py`
  - CPU CFG metric checks for input-feature and cond-output modes
- No GPU generation or PSNR rerun was launched after this scheduling optimization.
