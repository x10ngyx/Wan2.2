# 2026-06-30 Gated MLP 5-Feature Retrain

## Context

The gated multi-feature MLP implementation had been trained with four cached
features:

```text
latent_pool temporal_var frame_diff_mean frame_diff_var
```

The intended cached feature set also includes `temporal_mean`. The user asked to
remove the superseded 4-feature trainings, update code/scripts/docs to use five
features, and rerun:

- sample split, 30 epochs
- row split, 30 epochs
- row split, 100 epochs

## Code And Documentation Changes

- Updated `adaptive_threshold_predictor/models.py`:
  - `DEFAULT_GATED_FEATURE_SETS` now uses:
    `latent_pool temporal_mean temporal_var frame_diff_mean frame_diff_var`
  - Verified the default gated model instantiates with all five features.
- Updated `adaptive_threshold_predictor/README.md`:
  - recommended gated MLP command now includes `temporal_mean`
  - output naming examples use `gated_5feature`
- Updated `reports/report_gated_multifeature_mlp_architecture.md`:
  - architecture report now describes five selected features
  - parameter count updated to `83,526`
  - report now includes completed 5-feature training results

## Cleanup

Removed superseded 4-feature result roots:

- `/hy-tmp/wan22_adaptive_threshold_mlp_gated_4feature_samplesplit_20260630_013006`
- `/hy-tmp/wan22_adaptive_threshold_mlp_gated_4feature_rowsplit_gpu_20260630_014852`
- `/hy-tmp/wan22_adaptive_threshold_mlp_gated_4feature_rowsplit_gpu_long100_20260630_015638`

Removed matching symlinks under `experiment_results/`.

## New Training Outputs

Timestamp: `20260630_021641`

- sample split 30:
  `/hy-tmp/wan22_adaptive_threshold_mlp_gated_5feature_samplesplit_20260630_021641`
- row split 30:
  `/hy-tmp/wan22_adaptive_threshold_mlp_gated_5feature_rowsplit_gpu_20260630_021641`
- row split 100:
  `/hy-tmp/wan22_adaptive_threshold_mlp_gated_5feature_rowsplit_gpu_long100_20260630_021641`

Each root contains:

- `commands/launch_train.sh`
- `logs/train.log`
- `metrics.json`
- `val_predictions.csv`
- best/final model outputs

Matching symlinks were created in `experiment_results/`.

## Results

| Run | Split | Epochs Run | Best Epoch | Best Val MAE | Best Val Loss | Final Val MAE |
|---|---|---:|---:|---:|---:|---:|
| gated 5-feature sample split 30 | sample | 12 / 30, early stopped | 7 | `0.1142528785` | `0.1050056725` | `0.1178081666` |
| gated 5-feature row split 30 | row | 30 / 30 | 30 | `0.0756697811` | `0.0667008773` | `0.0756697811` |
| gated 5-feature row split 100 | row | 100 / 100 | 98 | `0.0601118673` | `0.0513865515` | `0.0601155481` |

Mean validation gate weights:

| Run | latent_pool | temporal_mean | temporal_var | frame_diff_mean | frame_diff_var |
|---|---:|---:|---:|---:|---:|
| sample split 30 | `0.5146` | `0.2312` | `0.0784` | `0.0689` | `0.1068` |
| row split 30 | `0.4112` | `0.2113` | `0.1137` | `0.1136` | `0.1502` |
| row split 100 | `0.3906` | `0.1825` | `0.1439` | `0.1116` | `0.1714` |

Comparison with removed 4-feature runs:

- sample split 30: old `0.1115179658`, new `0.1142528785`
- row split 30: old `0.0759376140`, new `0.0756697811`
- row split 100: old `0.0612696086`, new `0.0601118673`

The 5-feature row-split runs improve slightly over the 4-feature runs, but the
gain is small. The sample-split run worsened. The row-split 100 result is still
well behind the MiniDiT row-split reference MAE `0.0380019387`.

## Validation

- Verified all five cached feature files exist in
  `/hy-tmp/wan22_adaptive_threshold_feature_cache_candidate_inverse_20260616_012409`.
- Verified feature-cache dataset loads all five feature tensors with dimension
  `128`.
- Verified 5-feature gated MLP trainable parameter count is `83,526`.
- Ran:
  `py_compile adaptive_threshold_predictor/models.py adaptive_threshold_predictor/data.py adaptive_threshold_predictor/train_gate.py`
- Ran `git diff --check`.
- Confirmed no `adaptive_threshold_predictor.train_gate` processes remained.
- Confirmed GPU was idle after training: `0 MiB / 81920 MiB`, no running GPU
  processes.

## Open Notes

- No commit was made.
- The workspace still has unrelated dirty/untracked files from previous adaptive
  predictor and SeaCache work; they were not reverted or modified for this
  cleanup except where listed above.
