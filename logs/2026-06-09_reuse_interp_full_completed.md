# 2026-06-09 reuse_interp full threshold experiment completed

- Full ZEUS-threshold reuse_interp experiment completed with no tmux session remaining and no failed records.
- Result root: /hy-tmp/wan22_zeus_threshold_reuse_interp_10prompt_5th_20260608_195427.
- Completed artifacts: 10 baseline videos, 50 threshold videos, 50 PSNR JSON files, 60 ffprobe JSON files, summary.csv, aggregate_by_threshold.csv/json.
- Settings: 10 prompts, fixed seed 42, t2v-A14B, 832x480, 45 frames, 50 dpm++ steps, offload_model=True, convert_model_dtype, block/cfg cache disabled, zeus_caching_mode=reuse_interp.
- Threshold aggregate results:
  - 0.005: overall speedup 1.119x, mean PSNR 26.208 dB, min frame PSNR 17.26 dB, reuse/recompute 53/447.
  - 0.02: overall speedup 1.576x, mean PSNR 20.955 dB, min frame PSNR 14.18 dB, reuse/recompute 183/317.
  - 0.08: overall speedup 2.259x, mean PSNR 20.932 dB, min frame PSNR 13.82 dB, reuse/recompute 280/220.
  - 0.20: overall speedup 2.614x, mean PSNR 20.985 dB, min frame PSNR 13.88 dB, reuse/recompute 310/190.
  - 0.60: overall speedup 2.754x, mean PSNR 21.020 dB, min frame PSNR 13.90 dB, reuse/recompute 320/180.
- Takeaway: 0.005 has the best PSNR but only modest speedup. Higher thresholds 0.08-0.60 cluster around 20.93-21.02 dB while speedup increases up to 2.754x; 0.60 is the fastest and not meaningfully worse than 0.20 by mean PSNR, but both are below the formal fixed ZEUS PSNR level.
