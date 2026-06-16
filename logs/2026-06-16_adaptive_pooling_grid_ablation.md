# 2026-06-16 Adaptive Pooling Grid Ablation

## Task

Test whether a larger pooling grid improves the adaptive timestep-threshold predictor.

## Run

Command:

```bash
/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m adaptive_threshold_predictor.run_grid_ablation \
  --out_root /hy-tmp/wan22_adaptive_threshold_grid_ablation_20260616_020314 \
  --grids 2x4x4 3x4x4 4x4x4 \
  --epochs 3 \
  --train_batch_size 256 \
  --cache_batch_size 8 \
  --cache_num_workers 4 \
  --device cuda \
  --dtype float32
```

Output root:

- `/hy-tmp/wan22_adaptive_threshold_grid_ablation_20260616_020314`

Summary files:

- `/hy-tmp/wan22_adaptive_threshold_grid_ablation_20260616_020314/grid_feature_ablation_best_summary.csv`
- `/hy-tmp/wan22_adaptive_threshold_grid_ablation_20260616_020314/grid_feature_ablation_best_summary.json`

## Setup

- Dataset mode: `candidate_inverse`
- Examples: `50000`
- Split: grouped by `sample_id`, `40000` train and `10000` validation examples
- Mandatory condition inputs: normalized timestep and normalized target PSNR
- Feature sets tested per grid: `latent_pool`, `temporal_mean`, `temporal_var`, `frame_diff_mean`, `frame_diff_var`

## Results

Best rows across the existing `2x2x2` baseline and the new larger-grid runs:

| rank | grid | feature | params | best_epoch | best_val_loss | best_val_mae |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | 2x2x2 | temporal_mean | 29377 | 2 | 0.012259 | 0.120107 |
| 2 | 2x4x4 | latent_pool | 53953 | 1 | 0.012434 | 0.118093 |
| 3 | 4x4x4 | latent_pool | 86721 | 1 | 0.012733 | 0.118652 |
| 4 | 2x2x2 | latent_pool | 29377 | 2 | 0.012755 | 0.116558 |
| 5 | 3x4x4 | temporal_mean | 70337 | 1 | 0.013236 | 0.124000 |

Full new-grid best-loss table is stored in the run root.

## Conclusion

Larger pooling grids did not improve validation loss in this 3-epoch single-split ablation. The closest larger-grid result is `2x4x4 latent_pool`, but it uses about 1.84x the parameters of the `2x2x2` baseline and still has slightly worse validation loss. Current loss-based default remains `2x2x2 temporal_mean`.

## Files Touched

- Updated `PROGRESS.md` with the grid-ablation outcome.
- Added this session log.
