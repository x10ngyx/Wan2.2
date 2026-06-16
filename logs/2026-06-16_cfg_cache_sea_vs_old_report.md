# 2026-06-16 Sea CFG Cache vs Original CFG Cache Report

- Read existing progress and report context for the CFG cache comparison.
- Confirmed the relevant result root: `/hy-tmp/wan22_cfg_cache_prompt01_50step_45f_480p_20260613_163243`.
- Checked `summary.csv`, `summary_with_cache.csv`, and `experiment_config.json`.
- Added `reports/report_cfg_cache_sea_vs_old_prompt01.md`, following the previous report style: experiment purpose, configuration, method settings, data source, result tables, and short conclusion.
- The report compares CFG-only original `threshold` candidates `0.02/0.03` with SeaCache-style `sea-threshold` candidates `0.10/0.20/0.30`.
- Did not run new generation, PSNR, or GPU jobs.
- Appended the report update to `PROGRESS.md`.
- Existing unrelated dirty worktree entries were left untouched.
