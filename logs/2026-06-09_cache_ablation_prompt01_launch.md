# 2026-06-09 cache ablation prompt 01 launch

- Started a single-process ablation run for prompt 01 to isolate which cache causes the quality drop.
- tmux session: `cache_ablation_p01_20260609_184625`
- Result root: `/hy-tmp/wan22_cache_ablation_prompt01_50step_45f_480p_20260609_184625`
- Launcher: `experiments/cache_ablation_prompt01_50step_45f_480p/run_tmux.sh`
- Runner: `experiments/cache_ablation_prompt01_50step_45f_480p/run_ablation.py`
- Candidate set: timestep only, block only, CFG only, timestep+block, timestep+CFG, block+CFG, all three.
- Shared baseline: reused existing prompt 01 baseline video and ffprobe from the ZEUS threshold experiment archive.
- Verified start: tmux session exists, runner log shows one shared WanT2V pipeline created, and the first candidate entered model loading.
