# 2026-06-23 Adaptive SeaCache Cache Lifecycle Fix

## Summary

Fixed the late-run OOM bug in adaptive SeaCache batch experiments. The root
cause was retaining historical adaptive/replay SeaCache cache instances across
candidates. Each cache instance owns runtime tensors, including SeaCache
`previous_feature`, `previous_residual`, and adaptive current-latent snapshots,
so `torch.cuda.empty_cache()` cannot release that memory while Python still
references the old cache objects.

## Changes

- `adaptive_seacache_wan22/cache.py`
  - removed historical `instances` lists from adaptive and replay factories;
  - added `clear_runtime_state()` to adaptive and replay cache classes;
  - added factory `clear_last_instance()` hooks to clear runtime state and drop
    the last cache reference after trace/summary extraction.
- Adaptive SeaCache runners
  - added `release_cache_factory()` helper;
  - call it after writing candidate traces and before `torch.cuda.empty_cache()`;
  - call it on exception paths as well;
  - covered Ali prompt12, train10, train15/test5, and overhead train5 runners.
- Documentation
  - updated `AGENTS.md` with the batch-runner cache lifecycle rule;
  - updated adaptive prototype and experiment READMEs;
  - corrected the overhead train5 README to describe online-vs-replay overhead.

## Validation

- Ran `/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m py_compile` for:
  - `adaptive_seacache_wan22/cache.py`
  - all four adaptive SeaCache `run_batch.py` files.
- Ran `bash -n` for all four adaptive SeaCache `run_tmux.sh` files.
- Ran `rg` for `self.instances` / `instances.append`; no historical cache
  retention remains in the adaptive SeaCache module or adaptive runners.

## Notes

- No GPU inference rerun was launched in this fix session.
- Existing unrelated `todo.md` newline-only diff was left untouched.
