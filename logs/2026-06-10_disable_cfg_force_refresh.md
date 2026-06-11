# 2026-06-10 disable CFG miss force refresh

- User decided to close forced recomputation on CFG cache miss before running three-cache threshold combination experiments.
- Confirmed `CFGCacheConfig.force_uncond_recompute_on_miss` and the CLI flag default are already `False`; the current hardcoded `True` was in `experiments/cache_ablation_prompt01_50step_45f_480p/run_ablation.py`.
- Updated the ablation/three-cache config template so `cfg_config()` passes `force_uncond_recompute_on_miss=False` and generated candidate config JSON records `"force_uncond_recompute_on_miss": false`.
- Left the CLI flag `--cfg_force_uncond_recompute_on_miss` available for explicit comparison experiments; normal threshold-combination runs should omit it.
- Validation passed:
  - `/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m py_compile experiments/cache_ablation_prompt01_50step_45f_480p/run_ablation.py generate.py wan/text2video.py wan/cfg_cache.py`
  - `/hy-tmp/miniconda3/envs/Wan2.2/bin/python generate.py --help`
  - `rg` found no remaining `force_uncond_recompute_on_miss=True` config in `generate.py`, `wan/`, or `experiments/`.
- Clarified comparison context: ablation `timestep_only` uses `zeus-threshold` threshold `0.02`, matching previous `zeus-threshold th_0p02 prompt_01` PSNR `18.606 dB`; it should not be compared as the fixed-cadence formal ZEUS result (`22.226 dB`, `1.983x` on prompt 01).
