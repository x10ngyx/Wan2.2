# SeaCache UniPC Ali-10 50-step 45f 480p

Runs timestep-cache-only SeaCache on `test_sets/ali_10/prompts.jsonl` with
`sample_solver=unipc`. Other generation/cache settings match the existing
SeaCache 50-step 45-frame 480p runs.

## Configuration

- task: `t2v-A14B`
- checkpoint: `/hy-tmp/models/Wan2.2-T2V-A14B`
- prompts: `test_sets/ali_10/prompts.jsonl`
- seed: `42`
- size: `832*480`
- frames: `45`
- steps: `50`
- solver: `unipc`
- timestep cache: `seacache`
- block cache: none
- CFG cache: none
- thresholds: `0.10 0.20 0.30 0.50`

The runner loads the WanT2V pipeline once, then generates all baselines and
SeaCache candidates sequentially in the same process.

## Validate Without GPU

```bash
/hy-tmp/miniconda3/envs/Wan2.2/bin/python \
  experiments/seacache_unipc_ali10_50step_45f_480p/run_batch.py \
  --cpu_validate
```

Expected work: `10` baselines and `40` SeaCache candidates.

## Launch

```bash
bash experiments/seacache_unipc_ali10_50step_45f_480p/run_tmux.sh
```

The default output root is:

```text
/hy-tmp/wan22_seacache_unipc_ali10_50step_45f_480p_<timestamp>
```
