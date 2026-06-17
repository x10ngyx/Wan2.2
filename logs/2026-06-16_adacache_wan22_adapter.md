# 2026-06-16 AdaCache Wan2.2 Isolated Adapter

## Summary

- Copied the official AdaCache source tree into `third_party/AdaCache` for repository-local version management.
- Removed nested `.git` metadata from the copied AdaCache repository.
- Implemented a Wan2.2 runtime adapter inside `third_party/AdaCache` only.
- Did not modify Wan2.2 main source files.

## Added Files

- `third_party/AdaCache/wan22_adacache/__init__.py`
- `third_party/AdaCache/wan22_adacache/adapter.py`
- `third_party/AdaCache/wan22_adacache/README.md`
- `third_party/AdaCache/run_wan22_adacache.py`

## Implementation Notes

- The adapter monkey-patches `WanModel.forward` and `WanAttentionBlock.forward` at runtime.
- It uses Wan2.2's existing explicit `(model_stage, branch)` key passed through `block_cache_key`.
- It keeps `cond` and `uncond` cache states separate.
- It clears the previous stage when Wan2.2 switches between high and low noise models, so the new stage starts cold.
- It follows official AdaCache scheduling:
  - all blocks cache residuals;
  - `cache_loc` only computes the shared cadence metric;
  - first, penultimate, and final denoising steps force recompute;
  - default codebook is `0.03:12,0.05:10,0.07:8,0.09:6,0.11:4,1.0:3`.
- Wan2.2 has no explicit Open-Sora spatial/temporal block split, so `t-attn`, `s-attn`, and `self-attn` map to self-attention residuals. `ca-mlp` maps to cross-attention plus FFN residuals.

## Validation

- Ran:
  - `/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m py_compile third_party/AdaCache/wan22_adacache/adapter.py third_party/AdaCache/run_wan22_adacache.py`
  - `/hy-tmp/miniconda3/envs/Wan2.2/bin/python third_party/AdaCache/run_wan22_adacache.py --block_cache adacache --adacache_res t-attn --help`
- Removed generated `__pycache__` directories afterward.
- No GPU generation smoke test was run.

## Notes

- Existing unrelated worktree changes were left untouched, including adaptive predictor logs and `taylorseer_wan22` status entries.
