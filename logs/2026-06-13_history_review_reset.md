# 2026-06-13 history review and reset

This log replaces the previous long `logs/` history. Before deletion, the old logs and `PROGRESS.md` were reviewed for durable handoff information. The old log files were then removed so `logs/` starts fresh from this review record.

## What was preserved

- Project goal: implement threshold-based timestep, block, and CFG cache methods; generate threshold-combination quality/speed data; train a small adaptive threshold predictor.
- Environment/resource facts:
  - workspace `/hy-tmp/work/Wan2.2`
  - conda env `/hy-tmp/miniconda3/envs/Wan2.2`
  - model weights `/hy-tmp/models/Wan2.2-T2V-A14B`
  - OpenVid prompt zip `/hy-tmp/openvid_100_wan22_prompts.zip`
  - A100 80GB GPU, driver `570.211.01`
  - `/hy-tmp` data disk
- Cache interface and composition rules:
  - CFG cache is the outer branch cache.
  - timestep cache runs before block cache for each actual cond/uncond branch.
  - cache keys must include stage and branch.
  - unified CLI aliases now exist for `--timestep_threshold`, `--block_threshold`, and `--bwcache_threshold`.
- Experiment defaults:
  - `t2v-A14B`
  - seed `42`
  - `832*480`
  - `45` frames
  - `50` DPM++ steps
  - compute-only timing
  - FFmpeg PSNR vs no-cache baseline
  - single-process batch runners for threshold sweeps.
- Important result roots and headline results:
  - fixed ZEUS 10-prompt run: `/hy-tmp/wan22_zeus_timestep_cache_50step_45f_480p_full_20260608_114307`, `1.986x`, `23.705 dB`.
  - ZEUS-threshold reference run: `/hy-tmp/wan22_zeus_threshold_reuse_interp_10prompt_5th_20260608_195427`.
  - three-cache 64-combo grid: `/hy-tmp/wan22_three_cache_threshold_grid_prompt01_50step_45f_480p_20260610_012518`.
  - cache ablation: `/hy-tmp/wan22_cache_ablation_prompt01_50step_45f_480p_20260609_184625`.
  - SeaCache prompt-01: `/hy-tmp/wan22_seacache_50step_45f_480p_20260611_191733`.
  - SeaCache prompt-02 dense/high-threshold runs: `/hy-tmp/wan22_seacache_prompt02_dense_20260611_204826` and `/hy-tmp/wan22_seacache_prompt02_highthr_20260612_000218`.
  - local OpenVid SeaCache prompts 76-100 launch root: `/hy-tmp/wan22_seacache_openvid100_50step_45f_480p_20260612_002814`.
- Report locations:
  - `reports/report.md`
  - `reports/report_main_experiments.md`
  - `reports/report_supplementary_experiments.md`
  - `reports/report_seacache_vs_zeus_threshold_prompt12.md`

## What was deleted

All previous files under `logs/` were deleted after review. They were mostly implementation session notes, launch/monitor records, retry notes, raw download summaries, and experiment snapshots already superseded by `PROGRESS.md`, reports, or archived experiment roots.

## New current state

- `PROGRESS.md` was reset to a concise initialization snapshot.
- `logs/` now contains this single review/reset log.
- Latest checks:
  - `tmux ls`: no server running.
  - `nvidia-smi`: A100 80GB visible and idle.
  - `/hy-tmp/models/Wan2.2-T2V-A14B`: about `118G`.
  - `/hy-tmp`: about `136G` free.

## Next recommended check

Inspect `/hy-tmp/wan22_seacache_openvid100_50step_45f_480p_20260612_002814` before assuming whether the local OpenVid prompts 76-100 SeaCache shard completed, because the old historical log ended while that tmux run was still active, and the current tmux check only says no session is running now.

## Follow-up documentation cleanup

- Simplified `AGENTS.md` after the reset by merging the old `代码与接口约定` and `数据与自适应阶段规划` sections into `项目目标`.
- Preserved the essential cache ordering, unified threshold CLI, dataset-row fields, and adaptive predictor constraints; no implementation logic was changed.
