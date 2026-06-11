# 2026-06-11 SeaCache Experiment Scripts

- User requested SeaCache effect experiments following `AGENTS.md` and the completed reference experiment `experiment_results/wan22_zeus_threshold_reuse_interp_10prompt_5th_20260608_195427`, with multi-candidate runs avoiding repeated checkpoint loading.
- Added `experiments/seacache_50step_45f_480p/run_batch.py`, a single-process batch runner that loads one WanT2V pipeline and sequentially runs selected SeaCache thresholds/prompts.
- Added `experiments/seacache_50step_45f_480p/summarize_results.py`, `run_tmux.sh`, and `README.md`.
- Defaults mirror the reference run: seed `42`, `832*480`, 45 frames, 50 dpm++ steps, block/CFG disabled, compute-only timing, FFmpeg PSNR vs no-cache baseline, and baseline artifacts reused from `/hy-tmp/wan22_zeus_threshold_reuse_interp_10prompt_5th_20260608_195427`.
- Default SeaCache thresholds are `0.05 0.10 0.20 0.30 0.50`.
- After the user noted that the default run was too many candidates, changed both `run_batch.py` and `run_tmux.sh` to default to `prompt_limit=1`, giving a prompt-01 pilot with 5 candidate generations. Full 10-prompt validation remains available with `PROMPT_LIMIT=0` or `--prompt_limit 0`.
- CPU validation passed: conda `py_compile` for runner/summarizer, `bash -n run_tmux.sh`, `run_batch.py --cpu_validate` reporting `prompt_count=1` and `expected_candidate_runs=5`, `run_batch.py --help`, `git diff --check`, and a temporary summarizer fixture.
- No GPU experiment was launched in this step.
