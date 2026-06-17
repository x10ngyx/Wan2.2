# 2026-06-16 AdaCache-DiT Method Review

## Summary

- Reviewed the official AdaCache-DiT/AdaCache repository for a planned Wan2.2-14B comparison-method reproduction.
- Cloned source to `/hy-tmp/work/AdaCache`.
- Read:
  - `/hy-tmp/work/AdaCache/README.md`
  - `/hy-tmp/work/AdaCache/configs/sample_adacache.py`
  - `/hy-tmp/work/AdaCache/configs/sample_adacache_moreg.py`
  - `/hy-tmp/work/AdaCache/inference.py`
  - `/hy-tmp/work/AdaCache/opensora_base/opensora/models/stdit/stdit3.py`

## Findings

- AdaCache is training-free and operates inside video DiT blocks.
- The released Open-Sora patch caches residual components, not final model output:
  - `t-attn`: temporal self-attention residual
  - `s-attn`: spatial self-attention residual
  - `ca-mlp`: cross-attention plus MLP residual
- The method recomputes selected residuals on adaptive intervals. After a recompute, it compares the new residual to the cached residual, normalizes by the previous interval, averages the metric across selected layers, and maps the metric to the next interval through a codebook.
- The default config uses `cache_res='t-attn'`, `cache_loc=[13]`, and 100-step codebook `{0.03: 12, 0.05: 10, 0.07: 8, 0.09: 6, 0.11: 4, 1.00: 3}`.
- First, penultimate, and final sampling steps are forced to recompute.
- MoReg optionally multiplies the residual-change metric by a temporal motion regularizer computed from frame-stride residual differences, plus a motion-gradient term.

## Changes

- Appended a short review note to `PROGRESS.md`.
- No Wan2.2 source code was changed.

## Next

- Map AdaCache residual-cache points onto Wan2.2 transformer block internals.
- Decide whether the first Wan2.2 reproduction should implement only default `t-attn` residual caching or also expose `s-attn` / `ca-mlp` equivalents if Wan2.2 block boundaries support them cleanly.
