# 2026-06-16 Sea Block Cache vs Original Block-Group Report

- Located original block-group results in `/hy-tmp/wan22_block_cache_only_50step_45f_480p_20260609_125436`.
- Located final SeaCache-style block-group results in `/hy-tmp/wan22_block_group_sea_full_prompt01_50step_45f_480p_20260614_235605`.
- Confirmed `/hy-tmp/wan22_block_group_sea_full_prompt01_50step_45f_480p_20260613_235449` was an early failed pilot with zero completed rows.
- Read `summary.csv`, `aggregate_by_method_threshold.csv`, `experiment_config.json`, runner logs, and Sea block candidate logs.
- Extracted Sea block total and high/low stage reuse counts from `Block-group cache summary` log records.
- Added `reports/report_block_cache_sea_vs_old_prompt01.md`, following the previous report style: experiment purpose, configuration, method settings, data source, result tables, and short conclusion.
- Did not run new generation, PSNR, or GPU jobs.
- Appended the report update to `PROGRESS.md`.
- Existing unrelated dirty worktree entries were left untouched.
