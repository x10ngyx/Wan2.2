# 2026-06-16 AdaCache VBench Smoke OOM

## Command

```bash
HF_HOME=/hy-tmp/hf-cache \
TRANSFORMERS_CACHE=/hy-tmp/hf-cache \
HF_HUB_CACHE=/hy-tmp/hf-cache/hub \
/hy-tmp/miniconda3/envs/Wan2.2/bin/python \
  experiments/adacache_vbench_50step_45f_480p/run_batch.py \
  --prompt_limit 1 \
  --exp_root /hy-tmp/wan22_adacache_vbench_smoke_20260616_1908 \
  --convert_model_dtype
```

## Result

- GPU mode was confirmed before launch; A100 80GB was idle.
- Baseline completed successfully:
  - video: `/hy-tmp/wan22_adacache_vbench_smoke_20260616_1908/baseline/vbench_every20_001.mp4`
  - compute elapsed: `533.455s`
  - baseline sampling memory observed around `44023 MiB`
- AdaCache candidate failed with CUDA OOM:
  - log: `/hy-tmp/wan22_adacache_vbench_smoke_20260616_1908/logs/adacache_vbench_every20_001.log`
  - failed record: `/hy-tmp/wan22_adacache_vbench_smoke_20260616_1908/failed/adacache_vbench_every20_001.txt`

## OOM Details

- Failure happened at step `0`, during the `uncond` branch self-attention call.
- `nvidia-smi` immediately before failure showed about `81001 MiB / 81920 MiB` used.
- Error message:
  - tried to allocate `732.00 MiB`
  - `716.94 MiB` free
  - process memory in use: `78.54 GiB`
  - PyTorch allocated: `73.15 GiB`
  - PyTorch reserved but unallocated: `4.90 GiB`

## Takeaway

The fully official-style AdaCache path, which caches residuals for every Wan2.2 block and keeps explicit `cond`/`uncond` branch states, does not fit on a single A100 80GB at the default `832*480`, `45f`, `50-step` setup. A practical comparison will need a reduced-memory variant such as selected-block-only caching, offloaded cache tensors, or a design that avoids simultaneous full-block cond/uncond residual residency.
