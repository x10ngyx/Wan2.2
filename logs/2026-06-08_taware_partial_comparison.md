# 2026-06-08 timestep-aware partial comparison

- Compared completed prompt 01 timestep_aware_interp results against the previous reuse_interp pilot while the batch run was still processing threshold 0.60.
- Completed thresholds at comparison time: 0.005, 0.02, 0.08, 0.20.
- Reuse/recompute timestep counts matched reuse_interp for all completed thresholds, so the observed differences come from the output reconstruction strategy rather than different skip counts.
- Results versus reuse_interp:
  - 0.005: PSNR improved from 26.954 to 28.196 dB, speedup essentially unchanged at about 1.11x, reuse/recompute 5/45.
  - 0.02: PSNR decreased from 18.606 to 18.075 dB, speedup slightly higher at 1.618x vs 1.607x, reuse/recompute 19/31.
  - 0.08: PSNR decreased from 18.873 to 17.782 dB, speedup slightly higher at 2.263x vs 2.251x, reuse/recompute 28/22.
  - 0.20: PSNR decreased from 18.900 to 17.666 dB, speedup slightly higher at 2.620x vs 2.599x, reuse/recompute 31/19.
- Interim takeaway: timestep_aware_interp helps the conservative threshold 0.005 but is worse than reuse_interp at more aggressive thresholds on prompt 01.
