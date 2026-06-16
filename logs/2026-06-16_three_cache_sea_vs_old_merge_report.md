# 2026-06-16 Three-cache Sea-style vs Old Merge Report Update

## Summary

- Added a representative-point comparison to `reports/report_three_cache_sea_threshold_grid_prompt01.md`.
- Compared Sea-style three-cache threshold grid with the old merge three-cache grid.
- No experiments were rerun.

## Sources

- Sea-style report: `reports/report_three_cache_sea_threshold_grid_prompt01.md`
- Sea-style result table: `/hy-tmp/wan22_three_cache_sea_prompt01_50step_45f_480p_20260614_005404/results/summary.csv`
- Old merge result table: `/hy-tmp/work/Wan2.2/experiment_results/wan22_three_cache_threshold_grid_prompt01_50step_45f_480p_20260610_012518/results/summary.csv`

## Added Comparison

- Highest finite PSNR point.
- Fastest candidates under PSNR thresholds:
  - `>=26 dB`
  - `>=24 dB`
  - `>=22 dB`
  - `>=20 dB`
  - `>=19 dB`
  - `>=18 dB`
  - `>=15 dB`
- Fastest finite candidate.

## Main Takeaways

- Sea-style has a much higher best finite PSNR: `37.465 dB` vs old merge `26.954 dB`.
- Sea-style dominates the `22-26 dB` quality range on speed/quality tradeoff.
- Old merge is slightly faster near `18 dB`, while Sea-style is slightly higher quality.
- Sea-style reaches higher speedup in the aggressive low-quality range.

## Files Changed

- `reports/report_three_cache_sea_threshold_grid_prompt01.md`
- `PROGRESS.md`
- `logs/2026-06-16_three_cache_sea_vs_old_merge_report.md`
