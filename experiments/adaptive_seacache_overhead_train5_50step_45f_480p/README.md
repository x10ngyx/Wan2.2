# Adaptive SeaCache Overhead Train5

Measures adaptive predictor overhead on 5 prompts selected from the adaptive
SeaCache train-prompt set. For each prompt and target PSNR value, the runner
executes two matching candidates:

- `online_adaptive`: runs the predictor at each SeaCache decision and records
  predictor timing in the per-step trace;
- `replay_threshold`: replays the recorded threshold trace without running the
  predictor, preserving the same reuse/recompute decisions when the trace is
  deterministic.

Existing no-cache OpenVid baselines are reused from:

```text
/hy-tmp/work/Wan2.2/experiment_results/openvid_100_seacache_trace_data
```

Each candidate archives the generated video, ffprobe JSON, FFmpeg PSNR against
the reused baseline, raw log, compute-time file, command record, and per-step
trace JSON/CSV. The summary table reports predictor timing for online runs and
online-vs-replay elapsed differences for overhead estimation.

Cache lifecycle requirement: both online adaptive and replay SeaCache cache
objects hold GPU tensors in runtime state. The runner must release the latest
cache instance after writing each trace/summary and before
`torch.cuda.empty_cache()`. Do not keep historical cache instances across
candidates or online/replay pairs.

Launch:

```bash
bash experiments/adaptive_seacache_overhead_train5_50step_45f_480p/run_tmux.sh
```

CPU validation:

```bash
/hy-tmp/miniconda3/envs/Wan2.2/bin/python \
  experiments/adaptive_seacache_overhead_train5_50step_45f_480p/run_batch.py \
  --cpu_validate
```
