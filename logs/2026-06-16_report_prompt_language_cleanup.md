# 2026-06-16 Report Prompt Language Cleanup

- Updated prompt summary rows in all four reader-facing reports to English, matching the original Ali prompt language:
  - `reports/report_seacache_vs_zeus_threshold_prompt12.md`
  - `reports/report_cfg_cache_sea_vs_old_prompt01.md`
  - `reports/report_block_cache_sea_vs_old_prompt01.md`
  - `reports/report_three_cache_sea_threshold_grid_prompt01.md`
- Verified with `rg` that the targeted Chinese prompt fragments no longer appear in those report prompt rows.
- Did not run new generation, PSNR, or GPU jobs.
- Appended this cleanup note to `PROGRESS.md`.
