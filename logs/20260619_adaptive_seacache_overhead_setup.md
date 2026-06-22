# 2026-06-19 Adaptive SeaCache Overhead Setup

## Summary

- Added predictor timing instrumentation to `adaptive_seacache_wan22/cache.py`.
- Added threshold replay cache support in the same standalone adaptive package.
- Created `experiments/adaptive_seacache_overhead_train5_50step_45f_480p/`.
- The overhead experiment will run after the active train15/test5 experiment completes.

## Design

- Prompt subset: first 5 train prompts from the train15/test5 selection:
  - `openvidhd_part1_085`
  - `openvidhd_part1_086`
  - `openvidhd_part1_059`
  - `openvidhd_part1_057`
  - `openvidhd_part1_016`
- Target PSNR values: `20`, `25`, `30`.
- For each prompt/target pair:
  - run online adaptive SeaCache with predictor timing enabled;
  - save per-step `predictor_elapsed_seconds`;
  - run replay SeaCache using the online threshold trace without predictor calls;
  - compare online vs replay compute elapsed and decision mismatch count.

## Validation

- `py_compile` passed for `adaptive_seacache_wan22/cache.py` and overhead `run_batch.py`.
- `bash -n` passed for overhead `run_tmux.sh`.
- CPU validation passed: all five selected prompts and baseline artifacts were found.

## Launch

- tmux session: `adaptive_seacache_overhead_train5`
- experiment root: `/hy-tmp/wan22_adaptive_seacache_overhead_train5_50step_45f_480p_20260619_143632`
- The tmux session is waiting for `adaptive_seacache_train15_test5` before starting GPU work.

## Notes

- The active train15/test5 tmux was not interrupted.
- No git commit was made.
