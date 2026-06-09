# 2026-06-09 Block-cache-only prompt 01 launch

- Committed current code/scripts/logs before launch: `42ee328 Add CFG and block cache experiment support`.
- Created result root `/hy-tmp/wan22_block_cache_only_50step_45f_480p_20260609_125436`.
- Reused prompt 01 baseline from `/hy-tmp/wan22_zeus_threshold_reuse_interp_10prompt_5th_20260608_195427` by copying `baseline/prompt_01.mp4`, baseline command, raw log, compute `.time`, and ffprobe JSON into the new result root.
- Started tmux session `block_cache_only_p01_125436` with `RESUME_EXISTING=True bash experiments/block_cache_only_50step_45f_480p/run_experiments.sh`.
- Initial pane confirmed baseline was skipped and `bwcache_th_0p05` started.
- Settings: block cache only, timestep/cache disabled, CFG cache disabled, prompt 01, seed 42, 832x480, 45 frames, 50 dpm++ steps, BWCache thresholds `0.05 0.15 0.30`, block-group thresholds `0.01 0.03 0.05`, block group size 5, pooled relative-L1.
