# 2026-06-16 Adaptive Predictor Condition-only Comparison

## Purpose

Compared condition-only controls against feature models under the same 30-epoch training setup and hidden dimensions. The goal was to check whether latent-derived features materially improve validation performance beyond timestep and target PSNR.

## Runs

Condition-only roots:

- `/hy-tmp/wan22_adaptive_threshold_condition_only_hdim64_20260616`
- `/hy-tmp/wan22_adaptive_threshold_condition_only_hdim16_20260616`
- `/hy-tmp/wan22_adaptive_threshold_condition_only_hdim8_20260616`

Stdout logs:

- `logs/2026-06-16_adaptive_condition_only_hdim64_train.log`
- `logs/2026-06-16_adaptive_condition_only_hdim16_train.log`
- `logs/2026-06-16_adaptive_condition_only_hdim8_train.log`

The runs used the same cached dataset/split, `30` epochs, batch size `256`, and target range as the feature long runs. The condition-only model uses timestep and target PSNR only.

## Best Validation Table

| model | hidden dim | params | best epoch | best val loss | best val MAE | last val loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `condition_only` | 64 | 12865 | 23 | 0.013834 | 0.125093 | 0.014155 |
| `temporal_mean` | 64 | 29377 | 1 | 0.012571 | 0.119011 | 0.019334 |
| `latent_pool` | 64 | 29377 | 3 | 0.012612 | 0.117695 | 0.023170 |
| `condition_only` | 16 | 913 | 26 | 0.014025 | 0.125755 | 0.014143 |
| `temporal_mean` | 16 | 3505 | 4 | 0.012254 | 0.119104 | 0.021120 |
| `latent_pool` | 16 | 3505 | 4 | 0.012473 | 0.118758 | 0.018201 |
| `condition_only` | 8 | 265 | 27 | 0.014505 | 0.128548 | 0.014542 |
| `temporal_mean` | 8 | 1433 | 6 | 0.013240 | 0.121107 | 0.019828 |
| `latent_pool` | 8 | 1433 | 4 | 0.013039 | 0.120766 | 0.016017 |

## Relative Improvement Over Same-capacity Condition-only

| hidden dim | feature | best val-loss improvement | best MAE improvement |
| ---: | --- | ---: | ---: |
| 64 | `temporal_mean` | 9.12% | 4.86% |
| 64 | `latent_pool` | 8.83% | 5.91% |
| 16 | `temporal_mean` | 12.63% | 5.29% |
| 16 | `latent_pool` | 11.07% | 5.56% |
| 8 | `temporal_mean` | 8.72% | 5.79% |
| 8 | `latent_pool` | 10.11% | 6.05% |

## Interpretation

Latent-derived features consistently improve best-checkpoint validation performance over timestep + target PSNR alone. The gains are not huge, but they are stable across the tested capacities: roughly `9-13%` best val-loss improvement and `5-6%` best MAE improvement.

Condition-only models are more stable late in training, while feature models overfit earlier. This means feature inputs carry useful information but increase sample-specific fitting risk. The fairest comparison should use best checkpoint / early stopping, not the final epoch.

Current single-split default candidate remains `hidden_dim=16` + `temporal_mean` by best validation loss.

## Files Changed

- Updated `PROGRESS.md`.
- Added this session log.
- No code changes.
