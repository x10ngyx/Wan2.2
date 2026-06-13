# 2026-06-14 Three Sea-Style Cache Prompt-01 Grid Launch

## What Changed

- Added `experiments/three_cache_sea_prompt01_50step_45f_480p/run_batch.py`.
- Added `experiments/three_cache_sea_prompt01_50step_45f_480p/run_tmux.sh`.
- Appended the launch status to `PROGRESS.md`.

## Experiment

- Result root: `/hy-tmp/wan22_three_cache_sea_prompt01_50step_45f_480p_20260614_005404`
- Symlink: `experiment_results/wan22_three_cache_sea_prompt01_50step_45f_480p_20260614_005404`
- tmux session: `three_cache_sea_p01_20260614_005404`
- Prompt: prompt 01 from `prompt.txt`
- Baseline root: `/hy-tmp/wan22_cache_ablation_prompt01_50step_45f_480p_20260609_184625`
- Baseline compute seconds: `522.603`
- Thresholds for each cache: `0.05 0.10 0.20 0.40 1.00`
- Total candidates: `125`

## Cache Setup

- timestep cache: SeaCache timestep cache
- block cache: block-group with `sea_full_rel_l1` metric and `accumulated` decision
- CFG cache: `sea-threshold`
- Composition order: CFG outermost; timestep cache per actual branch; block-group cache only after timestep miss.

## Validation

- Passed Python compile:
  `/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m py_compile experiments/three_cache_sea_prompt01_50step_45f_480p/run_batch.py wan/timestep_cache.py wan/block_group_cache.py wan/cfg_cache.py wan/text2video.py generate.py`
- Passed CPU validation:
  `/hy-tmp/miniconda3/envs/Wan2.2/bin/python experiments/three_cache_sea_prompt01_50step_45f_480p/run_batch.py --cpu_validate`
- Passed shell syntax:
  `bash -n experiments/three_cache_sea_prompt01_50step_45f_480p/run_tmux.sh`

## Launch Status

- `bash experiments/three_cache_sea_prompt01_50step_45f_480p/run_tmux.sh` launched successfully.
- First candidate `sea_ts_0p05__sea_bg_0p05__sea_cfg_0p05` entered sampling.
- Launch GPU check showed about `63107 MiB` used and `100%` utilization during sampling.

## Follow-Up

- Monitor with:
  `tmux attach -t three_cache_sea_p01_20260614_005404`
- Check progress with:
  `tail -f /hy-tmp/wan22_three_cache_sea_prompt01_50step_45f_480p_20260614_005404/runner.log`
- Completed rows should appear incrementally in:
  `/hy-tmp/wan22_three_cache_sea_prompt01_50step_45f_480p_20260614_005404/results/summary.csv`
- If high-threshold block sea-full combinations OOM, inspect `failed/` and resume with the default `RESUME_EXISTING=True`.
