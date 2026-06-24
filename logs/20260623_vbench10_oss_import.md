# 2026-06-23 VBench10 OSS Import

## What Was Done

- Logged into Hengyuan OSS using the user-provided account.
- Listed `oss://datasets/` and found the requested VBench10 report/script and full-result archives plus checksum archives.
- Downloaded these files to `/hy-tmp/oss_downloads_20260623/`:
  - `wan22_vbench10_reports_and_experiment_scripts_20260623.tar.gz`
  - `wan22_vbench10_reports_and_experiment_scripts_20260623_checksum.tar.gz`
  - `wan22_vbench10_three_cache_full_merge_and_timestep_only_full_20260623.tar.gz`
  - `wan22_vbench10_three_cache_full_merge_and_timestep_only_full_20260623_checksum.tar.gz`
- Verified SHA256 checksums for both main archives.

## Imported Files

Reports added under `reports/`:

- `report_three_cache_sea_vbench10_merge.md`
- `report_timestep_only_seacache_vbench10.md`
- `report_compare_three_cache_merge_vs_timestep_only_vbench10.md`

Experiment scripts added under `experiments/`:

- `seacache_vbench10_50step_45f_480p/`
- `three_cache_sea_vbench10_50step_45f_480p/`

Full results extracted under:

- `/hy-tmp/wan22_vbench10_three_cache_full_merge_and_timestep_only_full_20260623/`

Workspace symlink added:

- `experiment_results/wan22_vbench10_three_cache_full_merge_and_timestep_only_full_20260623`

## Verification

- Reports/scripts archive SHA256: `a95282ae6e567e69b033ffcbc46abb657a4a8f7feaaee9fd87e232e20612b9bc`
- Full results archive SHA256: `1211cb89b75b75b340a9e53910db52f9e1c981b1e12694b400fc66ce03be4a84`
- Extracted full result counts:
  - `760` mp4 files
  - `26` csv files
  - `1532` json files
  - `0` failed files

## Notes

- The full result package contains `three_cache_sea_vbench10_full/` and `timestep_only_seacache_vbench10_full/`.
- Existing unrelated dirty worktree changes were left untouched.
- Download archives remain in `/hy-tmp/oss_downloads_20260623/`; remove them later if disk space is needed.
