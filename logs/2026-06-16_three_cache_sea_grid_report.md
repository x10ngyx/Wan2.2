# 2026-06-16 Sea-Style Three-Cache Grid Report

- Located completed sea-style three-cache grid at `/hy-tmp/wan22_three_cache_sea_prompt01_50step_45f_480p_20260614_005404`.
- Read `experiment_config.json`, `runner.log`, and `results/summary.csv`.
- Computed report summaries from the existing CSV only:
  - completed/finite/Infinity counts
  - fastest and best-PSNR candidates
  - fastest candidates by PSNR target
  - representative combinations
  - aggregate trends by timestep/block/CFG threshold
  - PSNR distribution bins
- Added `reports/report_three_cache_sea_threshold_grid_prompt01.md`, following the prior report style: experiment purpose, configuration, data source, result tables, and short conclusion.
- Added the complete 125-row result table to the report after user request, using compact columns from `results/summary.csv`.
- Did not run new generation, PSNR, or GPU jobs.
- Appended the report update to `PROGRESS.md`.
- Existing unrelated dirty worktree entries were left untouched.
