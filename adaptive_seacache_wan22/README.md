# Adaptive SeaCache Wan2.2 Prototype

This directory contains a standalone prototype for using the adaptive threshold
predictor during Wan2.2 T2V inference without changing the main Wan2.2 code.

The intended inference mode is timestep-only SeaCache:

1. Run Wan2.2 T2V normally with `--timestep_cache seacache`.
2. At every SeaCache block-reuse decision, predict a threshold from the current
   timestep feature and a user-provided `--target_psnr`.
3. Use that threshold for the current step only.

The implementation monkey-patches `wan.text2video.SeaCacheTimestepCache` and
wraps `WanModel.forward` in the current Python process, then calls the existing
`generate.py` pipeline. The forward wrapper only passes the current raw latent
to the adaptive cache so that online feature extraction matches training. No
files under `wan/` or the main `generate.py` are modified.

## Recommended First Test

Use the current best single-split model:

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

The Wan log line `Timestep cache summary: ...` includes
`adaptive_threshold_path`, plus min/max/mean predicted thresholds per
`(model_stage, branch)` key.

## Notes

- This prototype supports `temporal_mean` and `latent_pool`, matching the cached
  feature extraction used during training.
- It is intentionally restricted to timestep-only T2V SeaCache. Keep
  `--block_cache none` and `--cfg_cache none` for the first test.
- `--seacache_threshold` is still required by the upstream CLI but only serves
  as the default config value; the adaptive cache overrides it for each step.
- Batch runners must not retain historical adaptive or replay SeaCache cache
  instances. SeaCache runtime state keeps GPU tensors such as
  `previous_feature`, `previous_residual`, and the current latent snapshot, so
  referenced old cache objects prevent `torch.cuda.empty_cache()` from freeing
  memory and can cause late-run OOM. After serializing each candidate's summary
  and trace, call the factory `clear_last_instance()` hook before starting the
  next candidate.
