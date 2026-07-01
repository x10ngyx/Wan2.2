# 2026-06-30 Sampling Solver Impact Report Update

Task: update the old ZEUS solver comparison into a sampling-solver impact report split into ZEUS and SeaCache sections, add the full VBench10 SeaCache `dpm++` vs `unipc` comparison, remove explanatory prose, check for errors, and rename the report.

Updated:
- deleted old path: `reports/report_zeus_solver_ali10_vbench10_comparison_20260624.md`
- added new path: `reports/report_sampling_solver_impact_zeus_seacache_20260630.md`
- `PROGRESS.md`

Report changes:
- Retitled the report to `Sampling Solver Impact on ZEUS and SeaCache`.
- Added a data-coverage section:
  - ZEUS ali-10 and VBench10 are complete for both solvers.
  - SeaCache VBench10 is complete for both solvers.
  - SeaCache ali-10 is not a complete full comparison because full `SeaCache + dpm++ + ali-10` is still missing.
- Kept the existing ZEUS aggregate and per-sample comparison tables.
- Added SeaCache VBench10 source artifact paths.
- Added SeaCache VBench10 aggregate comparison for thresholds `0.10`, `0.20`, `0.30`, and `0.50`.
- Added SeaCache VBench10 per-sample comparison tables for each overlapping threshold.
- Removed explanatory/takeaway text and kept the report table-focused.

Validation:
- Verified SeaCache source CSVs exist and row counts match: `dpm++` aggregate 10 thresholds, `unipc` aggregate 4 thresholds, both summaries have 10 VBench10 samples and 10 rows for each overlapping threshold.
- Recomputed overlapping SeaCache aggregate deltas from source CSVs and confirmed the report values.
- `git diff --check -- reports/report_sampling_solver_impact_zeus_seacache_20260630.md reports/report_zeus_solver_ali10_vbench10_comparison_20260624.md` passed.

No experiment jobs were launched.
