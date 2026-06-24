# 2026-06-23 AdaCache VBench10 OSS Import

## Source

Downloaded from OSS:

- `oss://datasets/adacache_reproduction_20260623/adacache_wan22_vbench10_reproduction_20260623.tar.gz`

Downloaded archive path:

- `/hy-tmp/oss_downloads_20260623/adacache_wan22_vbench10_reproduction_20260623.tar.gz`

No separate checksum object was listed in the OSS directory. Local SHA256:

- `db66e61a1180f1c91e0b3b0643bdd07baf813b3c8dbb2a293164a61932b7b7a5`

## Imported Archive

Extracted full archive to:

- `/hy-tmp/adacache_wan22_vbench10_reproduction_20260623/`

Added symlink:

- `experiment_results/adacache_wan22_vbench10_reproduction_20260623`

The extracted archive contains:

- `report/`
- `experiment_scripts/`
- `experiment_results/`

## Repository Copies

Report imported to:

- `reports/adacache_vbench10_reproduction_report.md`

Experiment scripts imported to:

- `experiments/adacache_wan22_vbench10_reproduction_20260623/README.md`
- `experiments/adacache_wan22_vbench10_reproduction_20260623/run_batch.py`
- `experiments/adacache_wan22_vbench10_reproduction_20260623/run_one_method.py`
- `experiments/adacache_wan22_vbench10_reproduction_20260623/run_tmux.sh`

Excluded from repository copy:

- `__pycache__/`
- `*.pyc`

## Verification

Lightweight extracted-result counts:

- `30` mp4 files
- `35` csv files
- `85` json files
- `0` failed files

The archived report states:

- 10/10 baseline videos completed.
- 10/10 AdaCache slow videos completed.
- 10/10 AdaCache fast videos completed.
- 20/20 PSNR JSON files completed.
- AdaCache slow aggregate: `1.545x` mean speedup, `23.561 dB` mean PSNR.
- AdaCache fast aggregate: `2.702x` mean speedup, `18.635 dB` mean PSNR.

## Notes

- Existing unrelated dirty worktree changes were left untouched.
- The original downloaded tarball remains in `/hy-tmp/oss_downloads_20260623/` for now.
