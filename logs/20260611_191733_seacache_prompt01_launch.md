# 2026-06-11 SeaCache prompt-01 launch

- Read the current `PROGRESS.md`, SeaCache runner, tmux launcher, README, and summarizer before launching.
- Confirmed GPU mode: `nvidia-smi` reported an idle NVIDIA A100 80GB PCIe with CUDA 12.8.
- Confirmed `/hy-tmp` had about `152G` free and no existing tmux server was running.
- Preflight validation passed:
  - `experiments/seacache_50step_45f_480p/run_batch.py --cpu_validate`
  - `bash -n experiments/seacache_50step_45f_480p/run_tmux.sh`
  - conda `py_compile` for SeaCache runner/summarizer and relevant cache/generation modules.
- Launched `bash experiments/seacache_50step_45f_480p/run_tmux.sh`.
- tmux session: `seacache_50step_45f_480p_20260611_191733`.
- Result root: `/hy-tmp/wan22_seacache_50step_45f_480p_20260611_191733`.
- Workspace symlink: `experiment_results/wan22_seacache_50step_45f_480p_20260611_191733`.
- Settings: prompt 01 only, thresholds `0.05 0.10 0.20 0.30 0.50`, seed `42`, `832*480`, 45 frames, 50 dpm++ steps, SeaCache timestep cache only, block cache disabled, CFG cache disabled, and reused no-cache baseline artifacts from `/hy-tmp/wan22_zeus_threshold_reuse_interp_10prompt_5th_20260608_195427`.
- Launch monitoring showed the runner active in tmux, result metadata files created, no failed records, and one-time WanT2V pipeline initialization/checkpoint loading in progress.
