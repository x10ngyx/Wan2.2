# Adaptive SeaCache Train15 Test5 OpenVid

Runs timestep-only adaptive SeaCache on 20 OpenVid prompts from the adaptive
predictor split:

- 15 prompts sampled from `train_sample_ids`;
- 5 prompts sampled from `val_sample_ids`, used here as the held-out test set.

The default random seed is `20260619`. Existing no-cache OpenVid baselines are
reused from:

```text
/hy-tmp/work/Wan2.2/experiment_results/openvid_100_seacache_trace_data
```

The runner does not regenerate baselines. It loads WanT2V once and sequentially
runs 60 adaptive candidates: 20 prompts times target PSNR values `20`, `25`,
and `30`.

Each candidate archives the generated video, ffprobe JSON, FFmpeg PSNR against
the reused baseline, raw log, compute-time file, command record, and per-step
adaptive trace JSON/CSV with predicted threshold, rel-L1, accumulated rel-L1,
and reuse/recompute decision.

Launch:

```bash
bash experiments/adaptive_seacache_train15_test5_50step_45f_480p/run_tmux.sh
```

CPU validation:

```bash
/hy-tmp/miniconda3/envs/Wan2.2/bin/python \
  experiments/adaptive_seacache_train15_test5_50step_45f_480p/run_batch.py \
  --cpu_validate
```
