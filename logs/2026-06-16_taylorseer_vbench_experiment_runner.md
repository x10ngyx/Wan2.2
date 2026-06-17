# 2026-06-16 TaylorSeer VBench Experiment Runner

## Summary

- Added a VBench batch experiment under `experiments/taylorseer_vbench_50step_45f_480p/`.
- Kept TaylorSeer method code under `third_party/taylorseer_wan22/`.
- The experiment runner loads the Wan2.2/TaylorSeer pipeline once per process and sequentially runs VBench prompts.

## Files Added

- `experiments/__init__.py`
- `experiments/taylorseer_vbench_50step_45f_480p/__init__.py`
- `experiments/taylorseer_vbench_50step_45f_480p/run_batch.py`
- `experiments/taylorseer_vbench_50step_45f_480p/run_tmux.sh`
- `experiments/taylorseer_vbench_50step_45f_480p/README.md`

## Experiment Defaults

- Prompt set: `test_sets/vbench_every20/prompts.jsonl`
- Task: `t2v-A14B`
- Checkpoint: `/hy-tmp/models/Wan2.2-T2V-A14B`
- Seed: `42`
- Size: `832*480`
- Frames: `45`
- Steps: `50`
- Solver: `dpm++`
- TaylorSeer:
  - `fresh_threshold=5`
  - `max_order=1`
  - `first_enhance=1`

## Archive Behavior

- Writes:
  - `videos/`
  - `logs/`
  - `commands/`
  - `ffprobe/`
  - `results/summary.csv`
  - `results/summary.json`
  - `failed/`
  - `experiment_config.json`
  - `launch.env`
  - `gpu.txt`
  - `runner.log`
- PSNR/speedup are left blank because this TaylorSeer-only VBench runner does not generate no-cache baselines.

## Validation

- Passed static compilation:
  - `/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m py_compile experiments/__init__.py experiments/taylorseer_vbench_50step_45f_480p/__init__.py experiments/taylorseer_vbench_50step_45f_480p/run_batch.py third_party/taylorseer_wan22/cache.py third_party/taylorseer_wan22/text2video.py third_party/taylorseer_wan22/generate_t2v.py`
- Passed CPU validation:
  - `/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m experiments.taylorseer_vbench_50step_45f_480p.run_batch --cpu_validate --prompt_limit 2`
  - Found `51` VBench sampled prompts and selected `2`.
- Passed help/import check:
  - `/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m experiments.taylorseer_vbench_50step_45f_480p.run_batch --help`

## Notes

- This machine has no visible GPU, so no generation was launched.
- Use the README smoke-test command on the target multi-GPU machine before running all 51 prompts.
