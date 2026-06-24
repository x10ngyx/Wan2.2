# 2026-06-24 ZEUS VBench10 Result Check

- Checked `/hy-tmp/wan22_zeus_vbench10_50step_45f_480p_20260624_003030`.
- Completion:
  - `10/10` fixed ZEUS videos and PSNR JSON files.
  - `50/50` ZEUS-threshold videos and PSNR JSON files.
  - `60` rows in `results/summary.csv`.
  - `0` failed files.
  - `70` ffprobe JSON files; all have `832x480` and `45` frames.
- Aggregate results:
  - fixed ZEUS: `2.021x`, mean PSNR `23.996 dB`, min PSNR `14.96 dB`.
  - ZEUS-threshold `0.005`: `1.129x`, mean PSNR `23.020 dB`, min PSNR `14.40 dB`.
  - ZEUS-threshold `0.02`: `1.604x`, mean PSNR `20.868 dB`, min PSNR `14.31 dB`.
  - ZEUS-threshold `0.08`: `2.282x`, mean PSNR `20.690 dB`, min PSNR `14.87 dB`.
  - ZEUS-threshold `0.20`: `2.648x`, mean PSNR `20.707 dB`, min PSNR `14.92 dB`.
  - ZEUS-threshold `0.60`: `2.793x`, mean PSNR `20.734 dB`, min PSNR `14.91 dB`.
- `nvidia-smi` returned `No devices were found` at result-check time, so the check was disk-only after run completion.
