# Wan2.2 AdaCache Runtime Adapter

This adapter implements an isolated Wan2.2 reproduction of the official
AdaCache method inside `third_party/AdaCache`. It does not edit Wan2.2 source
files such as `generate.py`, `wan/text2video.py`, or `wan/modules/model.py`.

## Method Match

- Every Wan2.2 transformer block stores residual caches, matching the official
  AdaCache implementation style.
- `cache_loc` is used only to compute the shared adaptive cadence metric.
- The adaptive cadence uses the official codebook mapping:
  `0.03:12,0.05:10,0.07:8,0.09:6,0.11:4,1.0:3`.
- First, penultimate, and final denoising steps force recompute.
- Cache state is keyed by explicit `(model_stage, branch)`, so `cond` and
  `uncond` never share state.
- When Wan2.2 switches between `high` and `low` models, the completed stage is
  cleared and the new stage starts cold.

Wan2.2 does not expose Open-Sora's separate spatial and temporal blocks.
Therefore `t-attn`, `s-attn`, and `self-attn` map to Wan2.2 block self-attention
residuals. `ca-mlp` maps to cross-attention plus FFN residuals.

## Usage

Run the wrapper instead of calling `generate.py` directly:

```bash
/hy-tmp/miniconda3/envs/Wan2.2/bin/python third_party/AdaCache/run_wan22_adacache.py \
  --block_cache adacache \
  --adacache_res t-attn \
  --adacache_loc 13 \
  --task t2v-A14B \
  --ckpt_dir /hy-tmp/models/Wan2.2-T2V-A14B \
  --size 832*480 \
  --frame_num 45 \
  --sample_steps 50 \
  --sample_solver dpm++ \
  --base_seed 42 \
  --offload_model True \
  --convert_model_dtype \
  --prompt "A cinematic shot of a mountain lake at sunrise." \
  --save_file /hy-tmp/wan22_adacache_smoke.mp4
```

Optional MoReg:

```bash
--adacache_moreg \
--adacache_moreg_strides 1 \
--adacache_moreg_steps 10,90 \
--adacache_moreg_hyp 0.385,8,1,2
```

The wrapper consumes AdaCache-only arguments, enables the runtime patch, removes
`--block_cache adacache` before delegating to Wan2.2's normal parser, then logs
an AdaCache summary after generation.
