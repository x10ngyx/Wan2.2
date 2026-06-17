# 2026-06-16 Adaptive Feature Ablation Hidden Dim 16

## Purpose

Tested whether the early overfitting seen in the 30-epoch `hidden_dim=64` adaptive predictor run was partly caused by excessive model capacity.

## Run

- Output root: `/hy-tmp/wan22_adaptive_threshold_feature_ablation_hdim16_20260616`
- Full stdout: `logs/2026-06-16_adaptive_feature_ablation_hdim16_train.log`
- Same data/split/cache setup as the `hidden_dim=64` long run.
- Changed only model capacity: `hidden_dim=16`.
- Parameters dropped from `29377` to `3505`.

Command:

```bash
/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m adaptive_threshold_predictor.run_feature_ablation \
  --out_root /hy-tmp/wan22_adaptive_threshold_feature_ablation_hdim16_20260616 \
  --cache_dir /hy-tmp/wan22_adaptive_threshold_feature_cache_candidate_inverse_20260616_012409 \
  --dataset_mode candidate_inverse \
  --epochs 30 \
  --batch_size 256 \
  --hidden_dim 16 \
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
| `temporal_mean` | 3505 | 4 | 0.012254 | 0.119388 | 3 | 0.119104 | 0.021120 | 0.151698 |
| `latent_pool` | 3505 | 4 | 0.012473 | 0.121571 | 3 | 0.118758 | 0.018201 | 0.144731 |

## Comparison To Hidden Dim 64

| feature | hdim64 best val loss | hdim16 best val loss | hdim64 epoch-30 val loss | hdim16 epoch-30 val loss |
| --- | ---: | ---: | ---: | ---: |
| `temporal_mean` | 0.012571 | 0.012254 | 0.019334 | 0.021120 |
| `latent_pool` | 0.012612 | 0.012473 | 0.023170 | 0.018201 |

## Interpretation

Reducing capacity improves best validation loss for both feature settings and reduces late overfitting for `latent_pool`. It does not eliminate early overfitting: validation loss still rises after the first few epochs while train loss continues decreasing. `hidden_dim=16` is a better short-term default candidate than `hidden_dim=64`, but early stopping and multi-seed validation are still needed.

## Files Changed

- Updated `PROGRESS.md`.
- Added this session log.
- No code changes.
