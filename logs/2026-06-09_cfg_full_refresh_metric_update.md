# 2026-06-09 CFG full-refresh and metric update

- Followed up on the three-cache merge trial where quality dropped to about 18 dB PSNR.
- Root cause hypothesis: CFG cache was making reuse decisions from cond predictions that may already have been approximated by timestep/block cache, and CFG miss refresh could store `cached_delta` from approximate `cond` plus recomputed `uncond`.
- Changed CFG miss force-refresh behavior:
  - `--cfg_force_uncond_recompute_on_miss` now forces a full CFG refresh.
  - On CFG miss with the flag enabled, both `cond` and `uncond` are recomputed with timestep/block reuse bypassed.
  - The cached CFG delta is refreshed from these recomputed branch outputs.
- Added `--cfg_metric`:
  - default: `timestep_modulated_input_rel_l1`
  - comparison option: `cond_output_rel_l1`
- The new default compares model-internal timestep-modulated input features instead of comparing cond model outputs. The feature is produced from WanModel patch embedding plus the first block's timestep-modulated norm feature, so the CFG reuse criterion is less dependent on already-approximated cond predictions.
- `timestep_modulated_latent_rel_l1` remains accepted as a compatibility alias, but it now uses the same model-internal feature path.
- Validation:
  - `/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m py_compile generate.py wan/text2video.py wan/cfg_cache.py wan/timestep_cache.py wan/block_cache.py wan/block_group_cache.py wan/modules/model.py`
  - `python -m py_compile generate.py wan/text2video.py wan/cfg_cache.py`
  - `/hy-tmp/miniconda3/envs/Wan2.2/bin/python generate.py --help`
  - CPU checks for the model-internal CFG feature path and `cond_output_rel_l1`
- No GPU generation or PSNR run was launched after this change.
