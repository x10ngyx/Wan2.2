# Fixed SeaCache Train15/Test5 OpenVid

Runs timestep-only fixed-threshold SeaCache on the same 20 OpenVid prompts used
by the adaptive SeaCache train15/test5 evaluation:

- 15 prompts sampled from `train_sample_ids`;
- 5 prompts sampled from `val_sample_ids`, used as the held-out test set.

The default random seed is `20260619`. Existing no-cache OpenVid baselines are
reused from:

```text
/hy-tmp/work/Wan2.2/experiment_results/openvid_100_seacache_trace_data
```

The runner does not regenerate baselines. It loads WanT2V once and sequentially
runs 80 fixed-threshold candidates: 20 prompts times SeaCache thresholds
`0.1`, `0.2`, `0.4`, and `0.6`.

Each candidate archives the generated video, ffprobe JSON, FFmpeg PSNR against
the reused baseline, raw log, compute-time file, command record, and summary
rows with speedup plus SeaCache reuse/recompute statistics parsed from the raw
log.

Launch:

```bash
bash experiments/seacache_train15_test5_50step_45f_480p/run_tmux.sh
```

CPU validation:

```bash
/hy-tmp/miniconda3/envs/Wan2.2/bin/python \
  experiments/seacache_train15_test5_50step_45f_480p/run_batch.py \
  --cpu_validate
```
