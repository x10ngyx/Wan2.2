# 2026-06-16 Adaptive Feature Ablation Long Training

## Purpose

Checked whether the short 3-epoch adaptive predictor ablation was misleading because `temporal_mean` and `latent_pool` had best validation metrics around epoch 2.

## Run

- Output root: `/hy-tmp/wan22_adaptive_threshold_feature_ablation_long_20260616`
- Full stdout: `logs/2026-06-16_adaptive_feature_ablation_long_train.log`
- Command:

```bash
/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m adaptive_threshold_predictor.run_feature_ablation \
  --out_root /hy-tmp/wan22_adaptive_threshold_feature_ablation_long_20260616 \
  --cache_dir /hy-tmp/wan22_adaptive_threshold_feature_cache_candidate_inverse_20260616_012409 \
  --dataset_mode candidate_inverse \
  --epochs 30 \
  --batch_size 256 \
  --hidden_dim 64 \
  --psnr_min 10 \
  --psnr_max 50 \
  --device cuda \
  --num_workers 0 \
  --save_val_predictions \
  --feature_sets temporal_mean latent_pool
```

## Result

| feature | best val-loss epoch | best val loss | best val MAE at that epoch | best MAE epoch | best val MAE | epoch-30 val loss | epoch-30 val MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `temporal_mean` | 1 | 0.012571 | 0.119452 | 2 | 0.119011 | 0.019334 | 0.143876 |
| `latent_pool` | 3 | 0.012612 | 0.121387 | 2 | 0.117695 | 0.023170 | 0.155156 |

## Interpretation

Both runs show continuing train-loss decrease with validation loss rising after the first few epochs. The epoch-2 short-run optimum is not evidence that the model needed more epochs; the current architecture/data split overfits early. `temporal_mean` remains the safer loss-based default. `latent_pool` still has better early MAE but less stable validation loss.

## Files Changed

- Updated `PROGRESS.md`.
- Added this session log.
- No code changes.
