# 2026-06-09 block-cache comparison summary

- Compared corrected BWCache and block-group prompt 01 results from `/hy-tmp/wan22_block_cache_only_50step_45f_480p_20260609_125436`.
- Baseline compute time: `522.603s`.
- Corrected aggregate results:
  - BWCache 0.05: `1.030x`, PSNR `28.895 dB`, reuse/recompute `4/96`.
  - BWCache 0.15: `1.597x`, PSNR `18.370 dB`, reuse/recompute `41/59`.
  - BWCache 0.30: `1.674x`, PSNR `17.632 dB`, reuse/recompute `44/56`.
  - block-group 0.01: `0.982x`, PSNR `Infinity`, reuse/recompute `0/800`.
  - block-group 0.03: `1.361x`, PSNR `19.396 dB`, reuse/recompute `244/556`.
  - block-group 0.05: `1.713x`, PSNR `19.491 dB`, reuse/recompute `374/426`.
- Takeaway: block-group is currently the stronger block-cache direction for the practical speed/quality frontier. `block-group 0.05` is both faster and higher PSNR than BWCache 0.15/0.30.
- Exception: BWCache 0.05 is the high-quality point but gives only marginal speedup.
- Reuse/recompute counts are not directly comparable across methods because BWCache counts branch-level calls while block-group counts per-group calls.
