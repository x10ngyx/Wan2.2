# 2026-06-16 Adaptive Feature Ablation Hidden Dim 8

## Purpose

Tested whether reducing the adaptive predictor hidden dimension below 16 further improves validation behavior.

## Run

- Output root: `/hy-tmp/wan22_adaptive_threshold_feature_ablation_hdim8_20260616`
- Full stdout: `logs/2026-06-16_adaptive_feature_ablation_hdim8_train.log`
- Same data/split/cache setup as the `hidden_dim=64` and `hidden_dim=16` long runs.
- Changed model capacity to `hidden_dim=8`.
- Parameters dropped to `1433`.

Command:

```bash
/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m adaptive_threshold_predictor.run_feature_ablation \
  --out_root /hy-tmp/wan22_adaptive_threshold_feature_ablation_hdim8_20260616 \
  --cache_dir /hy-tmp/wan22_adaptive_threshold_feature_cache_candidate_inverse_20260616_012409 \
  --dataset_mode candidate_inverse \
  --epochs 30 \
  --batch_size 256 \
  --hidden_dim 8 \
  --psnr_min 10 \
  --psnr_max 50 \
  --device cuda \
  --num_workers 0 \
  --save_val_predictions \
  --feature_sets temporal_mean latent_pool
```

## Result

| feature | params | best val-loss epoch | best val loss | best val MAE at that epoch | best MAE epoch | best val MAE | epoch-30 val loss | epoch-30 val MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `temporal_mean` | 1433 | 6 | 0.013240 | 0.122773 | 8 | 0.121107 | 0.019828 | 0.144768 |
| `latent_pool` | 1433 | 4 | 0.013039 | 0.120766 | 4 | 0.120766 | 0.016017 | 0.132199 |

## Capacity Comparison

| hidden dim | params | `temporal_mean` best val loss | `latent_pool` best val loss |
| ---: | ---: | ---: | ---: |
| 64 | 29377 | 0.012571 | 0.012612 |
| 16 | 3505 | 0.012254 | 0.012473 |
| 8 | 1433 | 0.013240 | 0.013039 |

## Interpretation

`hidden_dim=8` further reduces late overfitting, but best validation loss gets worse than `hidden_dim=16`. Among the tested capacities, `hidden_dim=16` is the current best tradeoff. `hidden_dim=8` appears under-capacity for best loss, while `hidden_dim=64` overfits faster.

## Files Changed

- Updated `PROGRESS.md`.
- Added this session log.
- No code changes.
