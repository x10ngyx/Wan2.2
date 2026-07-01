# 2026-06-30 Gated MLP Row-Split Long100 Check

## Context

User asked whether the 30-epoch row-split gated four-feature MLP training was too short.

## Experiment

Ran the existing long confirmation training on GPU:

- Result root: `/hy-tmp/wan22_adaptive_threshold_mlp_gated_4feature_rowsplit_gpu_long100_20260630_015638`
- Symlink: `experiment_results/wan22_adaptive_threshold_mlp_gated_4feature_rowsplit_gpu_long100_20260630_015638`
- Feature cache: `/hy-tmp/wan22_adaptive_threshold_feature_cache_candidate_inverse_20260616_012409`
- Model: gated per-feature MLP with four features: `latent_pool`, `temporal_var`, `frame_diff_mean`, `frame_diff_var`
- Split: `row`
- Epochs: `100`
- Early-stop patience: `20`
- Device: `cuda`
- Parameters: `71,045`

## Results

- Completed all 100 epochs, no early stop.
- Best epoch: `97`
- Best validation MAE: `0.06126960859298706`
- Best validation loss: `0.052473511600494384`
- Train MAE at best epoch: `0.0634434845218435`
- Final epoch validation MAE: `0.06131729580387473`

Training checkpoints from this run:

- Epoch 30 validation MAE: `0.07319095213487745`
- Epoch 60 validation MAE: `0.06421787564083933`
- Epoch 80 validation MAE: `0.062042853160202506`
- Epoch 97 validation MAE: `0.06126960859298706`
- Epoch 100 validation MAE: `0.06131729580387473`

Gate means from `val_predictions.csv`:

- `latent_pool`: `0.48142299596108495`
- `temporal_var`: `0.16022270301587413`
- `frame_diff_mean`: `0.1453700610753149`
- `frame_diff_var`: `0.21298423908762634`

Best-epoch validation MAE by threshold:

- `0.10`: `0.013184659295912945`
- `0.15`: `0.028949345171692883`
- `0.20`: `0.03718816897174193`
- `0.25`: `0.05230508585996933`
- `0.30`: `0.057494931257026705`
- `0.40`: `0.08477556575241128`
- `0.50`: `0.07241613479008428`
- `0.60`: `0.06952267614009944`
- `0.70`: `0.06572135190169016`
- `0.80`: `0.13376738495910118`

Best-epoch validation MAE by step range:

- `step_00_09`: `0.08214041604310746`
- `step_10_39`: `0.056478400651589274`
- `step_40_49`: `0.05446347331869537`

## Interpretation

The 30-epoch setting was materially short for row split. The earlier 30-epoch row-split gated MLP best validation MAE was `0.07593761396706104`, while the 100-epoch confirmation improved to `0.06126960859298706`.

However, this still does not match the row-split MiniDiT reference best validation MAE `0.03800193872973323`, so epoch budget alone is not the whole explanation for the gap.

## Files Updated

- Appended result summary to `PROGRESS.md`.
- Added this session log.

## Validation

- Read `metrics.json` and `val_predictions.csv`.
- Confirmed the long training completed and wrote `saved: /hy-tmp/wan22_adaptive_threshold_mlp_gated_4feature_rowsplit_gpu_long100_20260630_015638`.
- No commit was made.
