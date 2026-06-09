# Block Cache Only: BWCache vs Block-Group

This experiment runs a one-prompt block-cache-only comparison for Wan2.2 T2V-A14B.

## Defaults

- task: `t2v-A14B`
- size: `832*480`
- frames: `45`
- steps: `50`
- solver: `dpm++`
- seed: `42`
- prompt: first prompt from `prompt.txt`
- timestep cache: `none`
- CFG cache: `none`
- baseline: no cache
- BWCache thresholds: `0.05 0.15 0.30`
- BWCache metric: `pooled_rel_l1`
- block-group thresholds: `0.01 0.03 0.05`
- block-group size: `5`
- block-group metric: `pooled_rel_l1`
- block-group protected window: `0.1` to `0.9` denoising progress
- block-group max consecutive reuse: `3`

## Run

```bash
bash experiments/block_cache_only_50step_45f_480p/run_experiments.sh
```

Useful overrides:

```bash
EXP_ROOT=/hy-tmp/wan22_block_cache_only_custom \
BWCACHE_THRESHOLDS="0.05 0.15 0.30" \
BWCACHE_METRIC=pooled_rel_l1 \
BLOCK_GROUP_THRESHOLDS="0.01 0.03 0.05" \
BLOCK_GROUP_START=0.1 \
BLOCK_GROUP_END=0.9 \
BLOCK_GROUP_MAX_REUSE=3 \
PROMPT_OFFSET=0 \
bash experiments/block_cache_only_50step_45f_480p/run_experiments.sh
```

## Outputs

The experiment root contains:

- `baseline/`
- `bwcache/th_*`
- `block_group/th_*`
- `commands/`
- `logs/`
- `ffprobe/`
- `psnr/`
- `results/summary.csv`
- `results/aggregate_by_method_threshold.csv`
- `results/aggregate_by_method_threshold.json`
- `failed/`

The script also creates a symlink under `experiment_results/`.
