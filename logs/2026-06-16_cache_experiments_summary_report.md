# 2026-06-16 Cache Experiments Summary Report

- Added `reports/report_cache_experiments_summary.md`.
- The report summarizes four existing reader-facing reports:
  - SeaCache vs ZEUS-threshold on Ali Prompt 1/2.
  - Sea CFG cache vs original CFG cache on Ali Prompt 1.
  - Sea block cache vs original block-group cache on Ali Prompt 1.
  - Sea-style three-cache threshold grid on Ali Prompt 1.
- Preserved experiment configurations, key complete result tables, and the complete 125-row three-cache grid result table.
- Removed secondary statistics from the combined report body, keeping only essential completion status, fastest/highest-quality/recommended points, and short conclusions.
- Verified the three-cache appendix contains 125 candidate rows and that targeted Chinese prompt fragments do not appear.
- Did not run new generation, PSNR, or GPU jobs.
