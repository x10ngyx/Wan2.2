# 2026-06-08 ZEUS-threshold prompt 01 pilot launch

- Updated experiments/zeus_threshold_50step_45f_480p/run_experiments.sh to support PROMPT_LIMIT and EXPECTED_THRESHOLD_COUNT while keeping the formal defaults at 10 prompts and 5 thresholds.
- Launched tmux session zeus_threshold_p01_162827 for a prompt 01 pilot.
- Result root: /hy-tmp/wan22_zeus_threshold_prompt01_7th_20260608_162827
- Workspace symlink: experiment_results/wan22_zeus_threshold_prompt01_7th_20260608_162827
- Command environment: PROMPT_LIMIT=1, EXPECTED_THRESHOLD_COUNT=7, THRESHOLDS=0.001 0.005 0.02 0.08 0.20 0.60 1.50.
- Shared settings remain t2v-A14B, 832*480, 45 frames, 50 steps, dpm++, fixed seed 42, offload_model=True, convert_model_dtype, block_cache none, cfg_cache none.
- Launch check: A100 80GB was idle; tmux pane showed baseline prompt 01 started and was loading T5 weights.
