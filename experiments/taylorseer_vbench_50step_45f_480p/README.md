# TaylorSeer VBench Batch Experiment

This experiment runs the standalone TaylorSeer Wan2.2 integration on the VBench
sampled prompt set:

```text
test_sets/vbench_every20/prompts.jsonl
```

The runner uses the project default T2V settings unless overridden:

- task: `t2v-A14B`
- checkpoint: `/hy-tmp/models/Wan2.2-T2V-A14B`
- seed: `42`
- size: `832*480`
- frame count: `45`
- sampling steps: `50`
- solver: `dpm++`
- dtype conversion: follows the TaylorSeer multi-GPU path; use FSDP + Ulysses on the target machine
- TaylorSeer defaults:
  - `fresh_threshold=5`
  - `max_order=1`
  - `first_enhance=1`
- baseline: no-cache WanT2V output with the same prompt, seed, shape, solver,
  steps, and guidance scale
- quality metric: FFmpeg PSNR against the no-cache baseline, using the same
  `compute_psnr.py` helper as the other cache experiments
- speed metric: baseline `inference_compute_elapsed_seconds` divided by
  TaylorSeer `inference_compute_elapsed_seconds`

## Batch Runner

Use:

```bash
torchrun --nproc_per_node=8 -m experiments.taylorseer_vbench_50step_45f_480p.run_batch \
  --ckpt_dir /hy-tmp/models/Wan2.2-T2V-A14B \
  --prompt_jsonl /hy-tmp/work/Wan2.2/test_sets/vbench_every20/prompts.jsonl \
  --exp_root /hy-tmp/wan22_taylorseer_vbench_50step_45f_480p_RUNID \
  --task t2v-A14B \
  --size 832*480 \
  --frame_num 45 \
  --sample_steps 50 \
  --sample_solver dpm++ \
  --base_seed 42 \
  --dit_fsdp \
  --t5_fsdp \
  --ulysses_size 8 \
  --taylorseer_fresh_threshold 5 \
  --taylorseer_max_order 1 \
  --taylorseer_first_enhance 1
```

The runner has two batch phases:

1. Load the no-cache WanT2V pipeline once per process, generate all missing
   baselines, run ffprobe, and record compute elapsed time.
2. Release the baseline pipeline, clear CUDA cache, then load the TaylorSeer
   pipeline once per process, generate all missing candidates, run ffprobe,
   compute FFmpeg PSNR against the baseline, and update the summary tables.

This keeps the batch-run property without keeping baseline and TaylorSeer model
instances resident at the same time. TaylorSeer cache state is reset at the
beginning of every sample generation, so prompts do not share cached module
outputs.

## Tmux Launcher

The tmux wrapper sets the usual `/hy-tmp` cache env vars and launches the batch
runner:

```bash
bash experiments/taylorseer_vbench_50step_45f_480p/run_tmux.sh
```

Useful environment overrides:

```bash
NPROC_PER_NODE=8
ULYSSES_SIZE=8
PROMPT_START=0
PROMPT_LIMIT=2
FRESH_THRESHOLD=5
MAX_ORDER=1
FIRST_ENHANCE=1
EXP_ROOT=/hy-tmp/wan22_taylorseer_vbench_smoke
SESSION=wan22_taylorseer_vbench_smoke
```

For a first multi-GPU smoke test on the target machine, use:

```bash
PROMPT_LIMIT=1 SAMPLE_STEPS=10 FRAME_NUM=9 \
EXP_ROOT=/hy-tmp/wan22_taylorseer_vbench_smoke \
SESSION=wan22_taylorseer_vbench_smoke \
bash experiments/taylorseer_vbench_50step_45f_480p/run_tmux.sh
```

Then inspect:

```bash
tail -f /hy-tmp/wan22_taylorseer_vbench_smoke/runner.log
cat /hy-tmp/wan22_taylorseer_vbench_smoke/gpu.txt
```

## CPU Validation

This does not import/load the model:

```bash
/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m experiments.taylorseer_vbench_50step_45f_480p.run_batch \
  --cpu_validate \
  --prompt_jsonl /hy-tmp/work/Wan2.2/test_sets/vbench_every20/prompts.jsonl \
  --prompt_limit 2
```

## Archive Layout

The experiment root contains:

```text
baseline/
videos/
logs/
commands/
ffprobe/
psnr/
results/
failed/
experiment_config.json
launch.env
gpu.txt
runner.log
```

`results/summary.csv` and `results/summary.json` include one row per completed
prompt, with sample id, generation parameters, TaylorSeer parameters, compute
elapsed time, baseline compute elapsed time, speedup, FFmpeg PSNR statistics,
ffprobe metadata, video/log/PSNR paths, and TaylorSeer reuse/recompute
statistics.

Key per-sample files:

```text
baseline/<sample_id>.mp4
videos/<sample_id>.mp4
logs/baseline_<sample_id>.log
logs/<sample_id>.log
ffprobe/baseline_<sample_id>.json
ffprobe/<sample_id>.json
psnr/<sample_id>.json
psnr/<sample_id>.log
commands/baseline_<sample_id>.sh
commands/<sample_id>.sh
```

Use `--resume_existing` or `RESUME_EXISTING=True` to skip samples whose required
artifacts already exist. A TaylorSeer sample is considered complete only after
the candidate video, candidate time file, candidate ffprobe JSON, and PSNR JSON
are present.
