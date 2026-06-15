# 2026-06-15 Three Sea-Style Cache Prompt-01 Grid Completion

## Checked Result Root

- Root: `/hy-tmp/wan22_three_cache_sea_prompt01_50step_45f_480p_20260614_005404`
- Symlink: `experiment_results/wan22_three_cache_sea_prompt01_50step_45f_480p_20260614_005404`

## Completion

- tmux has exited.
- GPU is idle.
- `failed/` is empty.
- `runner.log` ends with `Completed experiment`.
- `results/summary.csv` contains `125` rows.
- Artifact counts: `125` videos, `125` candidate ffprobe JSON files, `125` candidate PSNR JSON files, `125` candidate logs, `125` command records.
- ffprobe validation: all candidate videos are `832x480`, `45` frames, `16 fps`, duration `2.812500s`.
- PSNR rows: `124` finite, `1` Infinity/all-perfect row.

## Key Results

- Fastest finite candidate: `sea_ts_1p00__sea_bg_1p00__sea_cfg_1p00`, `5.644x`, PSNR `11.914 dB`.
- Best finite PSNR candidate: `sea_ts_0p05__sea_bg_0p10__sea_cfg_0p05`, `0.987x`, PSNR `37.465 dB`.
- Best speed at PSNR `>=35 dB`: `sea_ts_0p10__sea_bg_0p10__sea_cfg_0p10`, `1.025x`, PSNR `36.747 dB`.
- Best speed at PSNR `>=26 dB`: `sea_ts_0p10__sea_bg_0p05__sea_cfg_0p20`, `1.208x`, PSNR `26.430 dB`.
- Best speed at PSNR `>=24 dB`: `sea_ts_0p20__sea_bg_0p20__sea_cfg_0p20`, `1.496x`, PSNR `24.898 dB`.
- Best speed at PSNR `>=19 dB`: `sea_ts_0p40__sea_bg_0p10__sea_cfg_1p00`, `2.845x`, PSNR `19.007 dB`.
- Best speed at PSNR `>=18 dB`: `sea_ts_1p00__sea_bg_0p05__sea_cfg_0p20`, `3.575x`, PSNR `18.233 dB`.
- Best speed at PSNR `>=15 dB`: `sea_ts_1p00__sea_bg_1p00__sea_cfg_0p20`, `4.873x`, PSNR `15.633 dB`.

## Takeaway

- The full `5*5*5` prompt-01 sea-style three-cache grid completed without OOM.
- Moderate thresholds around `0.10-0.20` are the useful high-quality region.
- Aggressive thresholds give large speedups but quickly degrade PSNR.
