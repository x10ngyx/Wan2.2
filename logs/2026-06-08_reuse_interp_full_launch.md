# 2026-06-08 reuse_interp full threshold experiment launch

- Launched the full ZEUS-threshold experiment using the original reuse_interp reconstruction strategy.
- Result root: /hy-tmp/wan22_zeus_threshold_reuse_interp_10prompt_5th_20260608_195427.
- Workspace symlink: experiment_results/wan22_zeus_threshold_reuse_interp_10prompt_5th_20260608_195427.
- tmux session: zeus_threshold_full_reuse_195427.
- Runner: single-process experiments/zeus_threshold_50step_45f_480p/run_batch.py.
- Thresholds: 0.005 0.02 0.08 0.20 0.60. Removed uninformative 0.001 and 1.50 from the full run.
- Seed/shape/settings: fixed seed 42, 10 prompts, t2v-A14B, 832x480, 45 frames, 50 steps, dpm++, offload_model=True, convert_model_dtype, block/cfg cache disabled.
- Seeded prompt 01 artifacts from the completed prompt 01 reuse_interp pilot into the full result root and launched with --resume_existing, so prompt 01 should be skipped and the run should continue from prompt 02 after one-time pipeline initialization.
