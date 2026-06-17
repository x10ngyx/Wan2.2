# 2026-06-16 AdaCache VBench Batch Runner

## Summary

- Added a VBench batch experiment runner for the isolated Wan2.2 AdaCache adapter.
- Location: `experiments/adacache_vbench_50step_45f_480p/`.
- The runner does not modify Wan2.2 main source files.

## Files

- `experiments/adacache_vbench_50step_45f_480p/run_batch.py`
- `experiments/adacache_vbench_50step_45f_480p/run_tmux.sh`
- `experiments/adacache_vbench_50step_45f_480p/README.md`

## Behavior

- Loads WanT2V once in a single Python process.
- Uses `test_sets/vbench_every20/prompts.jsonl`.
- For each prompt, runs:
  - no-cache baseline;
  - AdaCache candidate.
- Uses the project default inference settings:
  - `t2v-A14B`
  - checkpoint `/hy-tmp/models/Wan2.2-T2V-A14B`
  - seed `42`
  - `832*480`
  - `45` frames
  - `50` steps
  - `dpm++`
  - offload enabled
  - dtype conversion enabled
- Archives videos, ffprobe JSON, PSNR JSON/logs, commands, raw logs, manifests, failed records, and result tables.

## Adapter Fix

- Added a runtime `enabled` switch to `third_party/AdaCache/wan22_adacache/adapter.py`.
- The runner installs the patch once, disables it for baseline generation, enables it for candidate generation, then disables and clears state afterward.
- This prevents later baselines in the same batch process from being affected by the monkey patch.

## Validation

- Ran `py_compile` for the adapter and runner.
- Ran CPU validation:
  - full prompt set: 51 prompts;
  - two-prompt subset.
- Removed generated `__pycache__` files.
- No GPU inference smoke test was launched.
