# OpenVid-100 ZEUS-Threshold Timestep Cache Experiment

This experiment creates prompt-threshold-generation-effect data for the 100
OpenVidHD prompts packaged at `/hy-tmp/openvid_100_wan22_prompts.zip`.

Default configuration is aligned with the earlier Wan2.2 T2V-A14B threshold
experiments:

- Model: Wan2.2 `t2v-A14B`
- Checkpoint: `/hy-tmp/models/Wan2.2-T2V-A14B`
- Prompts: `openvid_100_wan22_prompts/dataset_100.jsonl`
- Seed: fixed `42`
- Resolution: `832*480`
- Frames: `45`
- Steps: `50`
- Solver: `dpm++`
- Timestep cache: `zeus-threshold`
- ZEUS threshold mode: `reuse_interp`, `acc_range=(8,47)`, `max_interval=6`, metric `rel_l1`
- Block cache: disabled
- CFG cache: disabled

Default thresholds:

```text
0.001 0.003 0.005 0.008 0.010 0.015 0.020 0.030 0.050 0.080
```

The full run generates 100 no-cache baselines plus 100 prompts x 10 thresholds,
for 1000 ZEUS-threshold candidate rows. The runner loads one WanT2V pipeline per
process and executes the selected baselines/candidates sequentially to avoid
repeated checkpoint shard loading.

Run CPU validation without importing `wan`:

```bash
/hy-tmp/miniconda3/envs/Wan2.2/bin/python \
  experiments/zeus_threshold_openvid100_50step_45f_480p/run_batch.py \
  --cpu_validate
```

Launch the full GPU run in tmux:

```bash
bash experiments/zeus_threshold_openvid100_50step_45f_480p/run_tmux.sh
```

Override thresholds or run a small subset:

```bash
THRESHOLDS="0.001 0.005 0.020" PROMPT_LIMIT=3 \
bash experiments/zeus_threshold_openvid100_50step_45f_480p/run_tmux.sh
```

The archive under `/hy-tmp` includes:

- baseline videos and ZEUS-threshold videos
- command records with full launch arguments and prompt text
- raw logs, compute-only time files, and GPU snapshot
- ffprobe JSON for every video
- FFmpeg PSNR JSON/logs against the same prompt/seed no-cache baseline
- selected OpenVid manifest JSONL/CSV and a copy of the source zip
- `results/summary.csv`
- `results/aggregate_by_threshold.csv`
- `results/aggregate_by_threshold_and_group.csv`
- failure records under `failed/`
