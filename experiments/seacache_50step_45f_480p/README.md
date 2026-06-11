# SeaCache Timestep Cache T2V Experiment

Configuration follows the completed ZEUS-threshold reuse-interp run:

- Model: Wan2.2 T2V-A14B
- Checkpoint: `/hy-tmp/models/Wan2.2-T2V-A14B`
- Prompts: `/hy-tmp/work/Wan2.2/prompt.txt` (prompt 01 by default for the pilot; set `PROMPT_LIMIT=0` for all 10)
- Baseline reference: `/hy-tmp/wan22_zeus_threshold_reuse_interp_10prompt_5th_20260608_195427`
- Seed: fixed `42`
- Steps: `50`
- Frames: `45`
- Resolution: `832*480`
- Solver: `dpm++`
- Method: `--timestep_cache seacache`
- Block cache: disabled
- CFG cache: disabled
- Default SeaCache thresholds: `0.10 0.15 0.20 0.25 0.30 0.40 0.50 0.60 0.70 0.80`

Run CPU validation only:

```bash
/hy-tmp/miniconda3/envs/Wan2.2/bin/python \
  experiments/seacache_50step_45f_480p/run_batch.py \
  --cpu_validate
```

Launch the default prompt-01 pilot in tmux:

```bash
bash experiments/seacache_50step_45f_480p/run_tmux.sh
```

Override thresholds or selected prompts:

```bash
THRESHOLDS="0.02 0.05 0.10 0.20" PROMPT_LIMIT=1 \
bash experiments/seacache_50step_45f_480p/run_tmux.sh
```

Run all 10 prompts after the pilot:

```bash
PROMPT_LIMIT=0 bash experiments/seacache_50step_45f_480p/run_tmux.sh
```

By default, the runner reuses the no-cache baseline artifacts from the reference experiment, because the prompt/seed/shape/solver settings match. To regenerate baseline videos in the same single-process run:

```bash
GENERATE_BASELINE=True bash experiments/seacache_50step_45f_480p/run_tmux.sh
```

The batch runner loads one WanT2V pipeline per process and then sequentially runs all selected prompts and thresholds. Outputs are archived under `/hy-tmp`, with a symlink under `experiment_results/`, including:

- copied or generated baseline videos
- SeaCache candidate videos
- command records
- raw run logs and compute-only timing files
- ffprobe JSON for every video
- FFmpeg PSNR JSON/logs against same prompt/seed baseline
- `results/summary.csv`
- `results/aggregate_by_threshold.csv`
- `results/aggregate_by_threshold.json`
- failure records under `failed/`
