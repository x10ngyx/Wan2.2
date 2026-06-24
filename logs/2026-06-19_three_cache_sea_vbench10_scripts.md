# 2026-06-19 Three Sea-Style Cache VBench10 Scripts

## Summary

Added a VBench10 three-cache sea-style grid experiment under:

```text
experiments/three_cache_sea_vbench10_50step_45f_480p/
```

The runner uses VBench10 prompts, 64 threshold combinations per prompt, and per-prompt result directories.

## Files Added

- `run_batch.py`
  - Single-process WanT2V batch runner.
  - Loads the pipeline once per shard.
  - Runs one baseline per selected VBench10 prompt.
  - Runs all 4*4*4 sea-style threshold combinations per prompt.
  - Writes prompt-local videos, logs, ffprobe JSON, PSNR JSON/logs, command records, failed records, and prompt summary files.
  - Also writes shard-level `results/summary.csv`, `results/summary.json`, `results/aggregate_by_candidate.csv`, and `results/aggregate_by_candidate.json`.
- `run_tmux.sh`
  - Starts the runner in a detached tmux session.
  - Supports `PROMPT_START` and `PROMPT_LIMIT` for machine sharding.
  - Defaults thresholds to `0.05 0.10 0.20 0.50` for timestep, block, and cfg caches.
  - Explicitly prepends Wan2.2 env binary directories to PATH so ffprobe and PSNR ffmpeg resolution are stable in tmux.
- `run_two_machine_plan.sh`
  - Prints the exact commands for machine A (`PROMPT_START=0`, `PROMPT_LIMIT=5`) and machine B (`PROMPT_START=5`, `PROMPT_LIMIT=5`).
- `merge_shards.py`
  - Merges shard `results/summary.csv` files and aggregates by candidate combination.

## Important Path Handling

`run_tmux.sh` chooses `/hy-tmp/env/Wan2.2/bin/ffprobe` when it exists; otherwise it falls back to `/hy-tmp/miniconda3/envs/Wan2.2/bin/ffprobe`.

`run_batch.py` passes a PATH with the selected Python directory, selected ffprobe directory, `/hy-tmp/env/Wan2.2/bin`, and `/hy-tmp/miniconda3/envs/Wan2.2/bin` into both `ffprobe` and PSNR subprocesses. This is to keep `compute_psnr.py` resolving the intended `ffmpeg` even when launched from tmux.

## Validation

The current machine has the unpacked runtime at `/hy-tmp/env/Wan2.2`; `/hy-tmp/miniconda3/envs/Wan2.2/bin/python` was not present.

Passed:

```bash
/hy-tmp/env/Wan2.2/bin/python -m py_compile \
  experiments/three_cache_sea_vbench10_50step_45f_480p/run_batch.py \
  experiments/three_cache_sea_vbench10_50step_45f_480p/merge_shards.py

bash -n \
  experiments/three_cache_sea_vbench10_50step_45f_480p/run_tmux.sh \
  experiments/three_cache_sea_vbench10_50step_45f_480p/run_two_machine_plan.sh

/hy-tmp/env/Wan2.2/bin/python \
  experiments/three_cache_sea_vbench10_50step_45f_480p/run_batch.py \
  --cpu_validate --prompt_start 0 --prompt_limit 5 \
  --ffprobe_bin /hy-tmp/env/Wan2.2/bin/ffprobe

/hy-tmp/env/Wan2.2/bin/python \
  experiments/three_cache_sea_vbench10_50step_45f_480p/run_batch.py \
  --cpu_validate --prompt_start 5 --prompt_limit 5 \
  --ffprobe_bin /hy-tmp/env/Wan2.2/bin/ffprobe
```

CPU validation showed 10 total VBench10 records, 5 prompts per shard, 64 candidates per prompt, and 320 candidates per shard.

## Not Run

No GPU inference was launched in this session.
