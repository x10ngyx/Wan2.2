# OpenVid-100 SeaCache Timestep Cache Experiment

This experiment creates prompt-threshold-generation-effect data for the current
machine's OpenVidHD shard, prompts 76-100, packaged at
`/hy-tmp/openvid_100_wan22_prompts.zip`.

Default configuration is aligned with the earlier Wan2.2 T2V-A14B experiments,
but uses SeaCache instead of ZEUS-threshold:

- Model: Wan2.2 `t2v-A14B`
- Checkpoint: `/hy-tmp/models/Wan2.2-T2V-A14B`
- Prompts: `openvid_100_wan22_prompts/dataset_100.jsonl`, zero-based
  `prompt_start=75`, `prompt_limit=25`
- Seed: fixed `42`
- Resolution: `832*480`
- Frames: `45`
- Steps: `50`
- Solver: `dpm++`
- Timestep cache: `seacache`
- SeaCache parameters: `power_exp=3.0`, `power_const=1.0`, `eps=1e-16`, `norm_mode=mean`
- Block cache: disabled
- CFG cache: disabled

Default thresholds:

```text
0.10 0.15 0.20 0.25 0.30 0.40 0.50 0.60 0.70 0.80
```

The default run on this machine generates 25 no-cache baselines plus 25 prompts
x 10 thresholds, for 250 SeaCache candidate rows. The runner loads one WanT2V
pipeline per process and executes the selected baselines/candidates
sequentially to avoid repeated checkpoint shard loading.

Run CPU validation without importing `wan`:

```bash
/hy-tmp/miniconda3/envs/Wan2.2/bin/python \
  experiments/seacache_openvid100_50step_45f_480p/run_batch.py \
  --cpu_validate
```

Launch this shard in tmux:

```bash
bash experiments/seacache_openvid100_50step_45f_480p/run_tmux.sh
```

Override thresholds or run a small subset:

```bash
THRESHOLDS="0.10 0.15 0.20" PROMPT_LIMIT=3 \
bash experiments/seacache_openvid100_50step_45f_480p/run_tmux.sh
```

The archive under `/hy-tmp` includes:

- baseline videos and SeaCache videos
- command records with full launch arguments and prompt text
- raw logs, compute-only time files, and GPU snapshot
- ffprobe JSON for every video
- FFmpeg PSNR JSON/logs against the same prompt/seed no-cache baseline
- selected OpenVid manifest JSONL/CSV and a copy of the source zip
- `results/summary.csv`
- `results/aggregate_by_threshold.csv`
- `results/aggregate_by_threshold_and_group.csv`
- failure records under `failed/`
