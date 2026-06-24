# Adaptive SeaCache Train-Split OpenVid-10

Runs timestep-only adaptive SeaCache on 10 randomly selected prompts from the
adaptive predictor train split. The default selection uses split seed `42` from
the predictor artifact and experiment sampling seed `20260619`.

The runner reuses the existing no-cache OpenVid baselines under:

```text
/hy-tmp/work/Wan2.2/experiment_results/openvid_100_seacache_trace_data
```

It does not regenerate baselines. It loads WanT2V once and sequentially runs the
30 adaptive candidates: 10 prompts times target PSNR values `20`, `25`, and
`30`.

For each candidate it archives:

- generated video;
- ffprobe JSON;
- FFmpeg PSNR against the matching reused no-cache baseline;
- raw run log and compute-time file;
- command record;
- per-step adaptive trace JSON and CSV containing predicted threshold,
  SeaCache rel-L1, accumulated rel-L1, and reuse/recompute decision.

Cache lifecycle requirement: because adaptive SeaCache stores GPU tensors in
per-candidate runtime state, the runner must release the latest cache instance
after writing trace/summary and before `torch.cuda.empty_cache()`. Do not keep a
history list of cache instances across candidates.

Launch:

```bash
bash experiments/adaptive_seacache_train10_50step_45f_480p/run_tmux.sh
```

Validate prompt selection and baseline availability without loading the model:

```bash
/hy-tmp/miniconda3/envs/Wan2.2/bin/python \
  experiments/adaptive_seacache_train10_50step_45f_480p/run_batch.py \
  --cpu_validate
```
