# 2026-06-16 Adaptive Control Baselines

## Task

Add and run baseline controls for the adaptive threshold predictor:

- no latent-derived feature input, only timestep and PSNR;
- feature branch kept but feature input replaced by random noise.

## Code Changes

- Added `ConditionOnlyAdaCacheGate` to `adaptive_threshold_predictor/models.py`.
- Added `--control_mode` to `adaptive_threshold_predictor/train_gate.py`:
  - `feature`
  - `condition_only`
  - `noise_feature`
- Added `--noise_seed` for reproducible noise-feature controls.

## Run

Cache:

- `/hy-tmp/wan22_adaptive_threshold_feature_cache_candidate_inverse_20260616_012409`

Output:

- `/hy-tmp/wan22_adaptive_threshold_controls_20260616`

Commands:

```bash
/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m adaptive_threshold_predictor.train_gate \
  --cache_dir /hy-tmp/wan22_adaptive_threshold_feature_cache_candidate_inverse_20260616_012409 \
  --dataset_mode candidate_inverse \
  --feature_set latent_pool \
  --control_mode condition_only \
  --epochs 3 \
  --batch_size 256 \
  --num_workers 0 \
  --device cuda \
  --save_val_predictions \
  --out_dir /hy-tmp/wan22_adaptive_threshold_controls_20260616/condition_only

/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m adaptive_threshold_predictor.train_gate \
  --cache_dir /hy-tmp/wan22_adaptive_threshold_feature_cache_candidate_inverse_20260616_012409 \
  --dataset_mode candidate_inverse \
  --feature_set latent_pool \
  --control_mode noise_feature \
  --noise_seed 1234 \
  --epochs 3 \
  --batch_size 256 \
  --num_workers 0 \
  --device cuda \
  --save_val_predictions \
  --out_dir /hy-tmp/wan22_adaptive_threshold_controls_20260616/noise_feature
```

## Results

| setting | params | best_epoch | best_val_loss | best_val_mae |
| --- | ---: | ---: | ---: | ---: |
| `noise_feature` | 29377 | 1 | 0.014648 | 0.131173 |
| `condition_only` | 12865 | 3 | 0.014652 | 0.128916 |

For comparison, the best existing `2x2x2` real-feature runs were:

| setting | params | best_epoch | best_val_loss | best_val_mae |
| --- | ---: | ---: | ---: | ---: |
| `temporal_mean` | 29377 | 2 | 0.012259 | 0.120107 |
| `latent_pool` | 29377 | 2 | 0.012755 | 0.116558 |

## Interpretation

Timestep and PSNR alone explain a large part of the candidate-inverse label, but real latent-derived features still give a meaningful validation-loss improvement over both no-feature controls.

The 3-epoch runs should not be called fully converged. Training loss continued decreasing, while validation loss for real features started rising after epoch 1 or 2. These runs are best treated as short early-stopping ablations.
