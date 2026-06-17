# 2026-06-16 Adaptive SeaCache Inference Prototype

## Purpose

Implemented a standalone path to test adaptive threshold prediction inside Wan2.2 T2V inference without changing the main Wan code. The requested behavior is timestep-only SeaCache where a target PSNR is provided and the predictor chooses a SeaCache threshold at every denoising step.

## Files Added

- `adaptive_seacache_wan22/__init__.py`
- `adaptive_seacache_wan22/cache.py`
- `adaptive_seacache_wan22/patch.py`
- `adaptive_seacache_wan22/generate_t2v.py`
- `adaptive_seacache_wan22/README.md`

## Design

The standalone CLI runs:

```bash
python -m adaptive_seacache_wan22.generate_t2v ...
```

It accepts normal Wan `generate.py` arguments plus:

- `--adaptive_gate_model`
- `--target_psnr`
- `--adaptive_feature_set`
- `--adaptive_hidden_dim`
- `--adaptive_feature_dim`
- `--adaptive_grid_size`
- `--adaptive_psnr_min`
- `--adaptive_psnr_max`
- `--adaptive_min_threshold`
- `--adaptive_max_threshold`

It monkey-patches `wan.text2video.SeaCacheTimestepCache` in the current Python process so the existing T2V pipeline constructs `AdaptiveSeaCacheTimestepCache`.

It also wraps `WanModel.forward` in the current process to pass the current raw latent to the adaptive cache. This is necessary because the native SeaCache hook receives transformer token features, while the trained adaptive predictor expects pooled raw latent features.

Online feature extraction mirrors `adaptive_threshold_predictor/build_feature_cache.py` for:

- `temporal_mean`
- `latent_pool`

## Validation

Passed:

```bash
/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m py_compile \
  adaptive_seacache_wan22/cache.py \
  adaptive_seacache_wan22/patch.py \
  adaptive_seacache_wan22/generate_t2v.py
```

Also passed a lightweight smoke test that:

- loaded `/hy-tmp/wan22_adaptive_threshold_feature_ablation_hdim16_20260616/temporal_mean/best_model.pt`;
- predicted thresholds from fake raw latents;
- ran `AdaptiveSeaCacheTimestepCache.should_reuse_blocks`;
- confirmed `summary()` includes `adaptive_threshold_path`.

CLI parse smoke passed for a realistic 832x480, 45-frame, 50-step T2V command.

## Suggested First Real Command

```bash
/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m adaptive_seacache_wan22.generate_t2v \
  --task t2v-A14B \
  --ckpt_dir /hy-tmp/models/Wan2.2-T2V-A14B \
  --size 832*480 \
  --frame_num 45 \
  --sample_steps 50 \
  --sample_solver dpm++ \
  --base_seed 42 \
  --offload_model True \
  --convert_model_dtype \
  --timestep_cache seacache \
  --block_cache none \
  --cfg_cache none \
  --adaptive_gate_model /hy-tmp/wan22_adaptive_threshold_feature_ablation_hdim16_20260616/temporal_mean/best_model.pt \
  --adaptive_feature_set temporal_mean \
  --adaptive_hidden_dim 16 \
  --target_psnr 25 \
  --prompt "YOUR PROMPT" \
  --save_file /hy-tmp/adaptive_seacache_test.mp4
```

## Notes

- No real Wan generation was launched in this session.
- For first quantitative comparison, run no-cache baseline, fixed SeaCache threshold baselines, and adaptive SeaCache with the same prompt/seed/shape, then compute FFmpeg PSNR and compute-only speedup.
