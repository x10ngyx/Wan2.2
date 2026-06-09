# 2026-06-08 timestep-aware prompt 01 batch launch

- Launched a single-process batch comparison for timestep_aware_interp on prompt 01.
- Result root: /hy-tmp/wan22_zeus_threshold_taware_prompt01_5th_20260608_191714.
- Workspace symlink: experiment_results/wan22_zeus_threshold_taware_prompt01_5th_20260608_191714.
- Reused baseline prompt 01 artifacts from /hy-tmp/wan22_zeus_threshold_prompt01_7th_20260608_162827, so baseline generation will be skipped with --resume_existing.
- Thresholds: 0.005 0.02 0.08 0.20 0.60. Dropped 0.001 and 1.50 as uninformative from the previous reuse_interp pilot.
- tmux session: zeus_taware_p01_191714. Initial pane confirmed the runner is creating one WanT2V pipeline and using zeus_caching_mode=timestep_aware_interp.
