# 2026-06-08 ZEUS-threshold prompt 01 pilot completed

- The shell-based prompt 01 seven-threshold pilot completed; no tmux sessions remained and no failed records were present.
- Result root: /hy-tmp/wan22_zeus_threshold_prompt01_7th_20260608_162827.
- This run used the original reuse_interp mode, not the newly added timestep_aware_interp mode.
- Baseline compute time: 522.603s. Videos validated at 832x480, 45 frames, 16 fps, 2.8125s.
- Aggregate by threshold:
  - 0.001: elapsed 522.711s, speedup 0.9998x, PSNR Infinity, reuse/recompute 0/50.
  - 0.005: elapsed 470.287s, speedup 1.1112x, mean PSNR 26.954 dB, reuse/recompute 5/45.
  - 0.02: elapsed 325.253s, speedup 1.6068x, mean PSNR 18.606 dB, reuse/recompute 19/31.
  - 0.08: elapsed 232.145s, speedup 2.2512x, mean PSNR 18.873 dB, reuse/recompute 28/22.
  - 0.20: elapsed 201.093s, speedup 2.5988x, mean PSNR 18.900 dB, reuse/recompute 31/19.
  - 0.60: elapsed 191.124s, speedup 2.7344x, mean PSNR 18.928 dB, reuse/recompute 32/18.
  - 1.50: elapsed 191.336s, speedup 2.7313x, mean PSNR 18.928 dB, reuse/recompute 32/18.
- Main takeaways: threshold 0.005 gives the only high-PSNR result but modest speedup; thresholds 0.02 and above collapse to roughly 18.6-18.9 dB while speedup rises; reuse saturates at 32/50 timesteps around threshold 0.60.
