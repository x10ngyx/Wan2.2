# 2026-06-09 block-cache-only results check

- Checked tmux: no active tmux server; `block_cache_only_p01_retry5_windowed` completed.
- Result root: `/hy-tmp/wan22_block_cache_only_50step_45f_480p_20260609_125436`.
- Active `failed/` directory has no records. Historical interrupted/OOM attempts remain under `failed_history/`.
- Completed artifacts present for baseline and all six candidates: videos, commands, raw logs, compute time files, ffprobe JSON, PSNR JSON/logs, summary CSV, and aggregate CSV/JSON.
- ffprobe validation: every video is `832x480`, `45` frames, `16 fps`, duration `2.812500s`.
- Baseline compute time: `522.603s`.
- Aggregate results:
  - BWCache 0.05: speedup `1.030x`, PSNR `28.895 dB`, reuse/recompute `4/96`.
  - BWCache 0.15: speedup `1.597x`, PSNR `18.370 dB`, reuse/recompute `41/59`.
  - BWCache 0.30: speedup `1.232x`, PSNR `16.824 dB`, reuse/recompute `21/79`.
  - block-group 0.01: speedup `0.982x`, PSNR `Infinity`, reuse/recompute `0/800`.
  - block-group 0.03: speedup `1.361x`, PSNR `19.396 dB`, reuse/recompute `244/556`.
  - block-group 0.05: speedup `1.713x`, PSNR `19.491 dB`, reuse/recompute `374/426`.
- Block-group threshold env files confirm `start=0.1`, `end=0.9`, `max_reuse=3`, `metric=pooled_rel_l1`.
