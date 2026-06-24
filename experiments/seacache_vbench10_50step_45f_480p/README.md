# SeaCache Vbench10 50-step 45f 480p

Runs timestep-cache-only SeaCache on the unified `Vbench10` prompt subset with
the same 10 thresholds used by the OpenVid SeaCache sweep.

## Configuration

- task: `t2v-A14B`
- checkpoint: `/hy-tmp/models/Wan2.2-T2V-A14B`
- prompts: `test_sets/Vbench10/prompts.jsonl`
- seed: `42`
- size: `832*480`
- frames: `45`
- steps: `50`
- solver: `dpm++`
- timestep cache: `seacache`
- block cache: none
- CFG cache: none
- thresholds: `0.10 0.15 0.20 0.25 0.30 0.40 0.50 0.60 0.70 0.80`

The runner loads the WanT2V pipeline once, then generates all baselines and
SeaCache candidates sequentially in the same process.

## Validate Without GPU

```bash
python experiments/seacache_vbench10_50step_45f_480p/run_batch.py --cpu_validate
```

Expected work: `10` baselines and `100` SeaCache candidates.

## Launch

```bash
bash experiments/seacache_vbench10_50step_45f_480p/run_tmux.sh
```

Preferred 2-GPU launch:

```bash
bash experiments/seacache_vbench10_50step_45f_480p/run_tmux_2gpu.sh
```

This starts two tmux sessions:

- GPU `0`: `prompt_start=0`, `prompt_limit=5`
- GPU `1`: `prompt_start=5`, `prompt_limit=5`

Each shard has its own experiment root and loads the pipeline once. After both
shards finish, merge result tables with:

```bash
python experiments/seacache_vbench10_50step_45f_480p/merge_shards.py \
  --parent-root /hy-tmp/wan22_seacache_vbench10_50step_45f_480p_<timestamp>
```

Optional smoke test:

```bash
PROMPT_LIMIT=1 THRESHOLDS="0.10 0.20" \
  EXP_ROOT=/hy-tmp/wan22_seacache_vbench10_smoke \
  SESSION=wan22_seacache_vbench10_smoke \
  bash experiments/seacache_vbench10_50step_45f_480p/run_tmux.sh
```

## Outputs

The default output root is:

```text
/hy-tmp/wan22_seacache_vbench10_50step_45f_480p_<timestamp>
```

Important files:

- `runner.log`: full batch log.
- `baseline/*.mp4`: no-cache baseline videos.
- `seacache/th_<threshold>/*.mp4`: candidate videos.
- `ffprobe/*.json`: generated video metadata.
- `psnr/th_<threshold>/*.json`: FFmpeg PSNR results.
- `results/summary.csv`: one row per candidate.
- `results/aggregate_by_threshold.csv`: aggregate speed/quality by threshold.
- `manifests/selected_vbench10_records.jsonl`: exact prompt records.

For 2-GPU launches, the parent root contains:

- `shard_gpu0_p000_004/`: first five prompts.
- `shard_gpu1_p005_009/`: last five prompts.
- `merged/summary.csv`: merged 100-candidate table after `merge_shards.py`.
- `merged/aggregate_by_threshold.csv`: merged threshold aggregate.
