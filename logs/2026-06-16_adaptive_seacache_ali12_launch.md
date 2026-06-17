# 2026-06-16 Adaptive SeaCache Ali Prompt 1-2 Launch

## Task

Launch adaptive SeaCache inference on Ali prompt 1 and 2 with target PSNR values `20`, `25`, and `30`, for comparison against prior timestep-only SeaCache results.

## Implementation

Added:

- `experiments/adaptive_seacache_ali_prompt12_50step_45f_480p/run_batch.py`
- `experiments/adaptive_seacache_ali_prompt12_50step_45f_480p/run_tmux.sh`
- `experiments/adaptive_seacache_ali_prompt12_50step_45f_480p/README.md`

Updated `adaptive_seacache_wan22/cache.py` so `summary()` includes per-step adaptive decision rows:

- `step_index`
- `predicted_threshold`
- `rel_l1`
- `accumulated_rel_l1`
- `decision`
- `force_recompute`

The batch runner writes these rows to per-candidate trace JSON and CSV files.

## Validation

Passed:

```bash
/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m py_compile \
  adaptive_seacache_wan22/cache.py \
  adaptive_seacache_wan22/patch.py \
  experiments/adaptive_seacache_ali_prompt12_50step_45f_480p/run_batch.py
```

CPU validation passed:

- prompts: `2`
- targets: `20 25 30`
- expected candidate runs: `6`
- baseline artifacts present

## Launch

Command:

```bash
bash experiments/adaptive_seacache_ali_prompt12_50step_45f_480p/run_tmux.sh adaptive_seacache_ali12
```

tmux session:

- `adaptive_seacache_ali12`

Experiment root:

- `/hy-tmp/wan22_adaptive_seacache_ali_prompt12_50step_45f_480p_20260616_165412`

Runner log:

- `/hy-tmp/wan22_adaptive_seacache_ali_prompt12_50step_45f_480p_20260616_165412/logs/runner.log`

Initial log check showed the adaptive gate loaded successfully. The run was still active when this log was written.

## Completion

The tmux session finished and exited.

- Completed candidates: `6/6`
- Failed files: `0`
- Summary CSV: `/hy-tmp/wan22_adaptive_seacache_ali_prompt12_50step_45f_480p_20260616_165412/results/summary.csv`
- Summary JSON: `/hy-tmp/wan22_adaptive_seacache_ali_prompt12_50step_45f_480p_20260616_165412/results/summary.json`

All generated and baseline videos passed ffprobe checks:

- resolution: `832x480`
- frames: `45`
- duration: `2.8125s`
- avg frame rate: `16/1`

| sample | target PSNR | speedup | measured PSNR | reuse | recompute | mean predicted threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ali_001` | 20 | 2.870x | 19.325 dB | 72 | 28 | 0.5330 |
| `ali_001` | 25 | 1.869x | 19.450 dB | 52 | 48 | 0.2819 |
| `ali_001` | 30 | 1.543x | 24.462 dB | 40 | 60 | 0.1809 |
| `ali_002` | 20 | 3.051x | 20.288 dB | 74 | 26 | 0.6230 |
| `ali_002` | 25 | 2.270x | 26.998 dB | 62 | 38 | 0.3777 |
| `ali_002` | 30 | 1.641x | 29.354 dB | 44 | 56 | 0.2126 |

Per-step traces are under:

- `/hy-tmp/wan22_adaptive_seacache_ali_prompt12_50step_45f_480p_20260616_165412/traces/`

Each trace row includes `predicted_threshold`, `rel_l1`, `accumulated_rel_l1`, and the reuse/recompute decision.

## Fixed SeaCache Comparison

Compared against previous fixed-threshold SeaCache runs:

- prompt 01: `/hy-tmp/wan22_seacache_50step_45f_480p_20260611_191733`
- prompt 02:
  - `/hy-tmp/wan22_seacache_prompt02_dense_20260611_204826`
  - `/hy-tmp/wan22_seacache_prompt02_highthr_20260612_000218`

Fixed threshold reference:

| prompt | threshold | speedup | PSNR |
| --- | ---: | ---: | ---: |
| 01 | 0.10 | 1.112x | 36.303 dB |
| 01 | 0.20 | 1.569x | 24.558 dB |
| 01 | 0.30 | 1.966x | 20.562 dB |
| 01 | 0.50 | 2.779x | 19.460 dB |
| 02 | 0.10 | 1.090x | 45.532 dB |
| 02 | 0.15 | 1.402x | 35.441 dB |
| 02 | 0.20 | 1.562x | 30.097 dB |
| 02 | 0.30 | 1.965x | 29.582 dB |
| 02 | 0.40 | 2.405x | 27.044 dB |
| 02 | 0.50 | 2.641x | 23.725 dB |
| 02 | 0.60 | 3.098x | 20.262 dB |
| 02 | 0.80 | 3.499x | 18.631 dB |

Nearest fixed-threshold comparison:

| adaptive point | adaptive speed / PSNR | nearest fixed by PSNR | nearest fixed by speed |
| --- | --- | --- | --- |
| p01 target 20 | 2.870x / 19.325 dB | th 0.50: 2.779x / 19.460 dB | th 0.50: 2.779x / 19.460 dB |
| p01 target 25 | 1.869x / 19.450 dB | th 0.50: 2.779x / 19.460 dB | th 0.30: 1.966x / 20.562 dB |
| p01 target 30 | 1.543x / 24.462 dB | th 0.20: 1.569x / 24.558 dB | th 0.20: 1.569x / 24.558 dB |
| p02 target 20 | 3.051x / 20.288 dB | th 0.60: 3.098x / 20.262 dB | th 0.60: 3.098x / 20.262 dB |
| p02 target 25 | 2.270x / 26.998 dB | th 0.40: 2.405x / 27.044 dB | th 0.40: 2.405x / 27.044 dB |
| p02 target 30 | 1.641x / 29.354 dB | th 0.30: 1.965x / 29.582 dB | th 0.20: 1.562x / 30.097 dB |

Interpretation:

- Adaptive thresholding follows the expected direction and lands near fixed SeaCache operating points.
- It does not yet beat the fixed-threshold sweep frontier on these two prompts.
- `p01 target 25` is the clearest bad point: nearly the same PSNR as fixed threshold `0.50`, but much slower, and worse than fixed threshold `0.30` in both PSNR and speed.
- Prompt 02 calibration is better, but fixed threshold `0.40` or `0.60` still has a small advantage at comparable operating points.
