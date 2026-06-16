# 2026-06-15 Adaptive Threshold Predictor Scaffold

## Summary

- Created `adaptive_threshold_predictor/` for prediction-network work, separate from generation runners.
- Inspected `/hy-tmp/openvid_100_seacache_trace_data/data`.
- Confirmed trace layout:
  - `data/tables/summary.csv`
  - `data/baseline/step_inputs/<sample_id>/meta.pt`
  - `data/baseline/step_inputs/<sample_id>/step_000.pt` ... `step_049.pt`
- Confirmed single-step latent shape is `[16, 12, 60, 104]` with dtype `float16` on disk.

## Files Added

- `adaptive_threshold_predictor/__init__.py`
- `adaptive_threshold_predictor/README.md`
- `adaptive_threshold_predictor/models.py`
- `adaptive_threshold_predictor/data.py`
- `adaptive_threshold_predictor/inspect_trace_data.py`
- `adaptive_threshold_predictor/train_gate.py`

## Model

- Added timestep-cache-only `ImprovedAdaCacheGate`.
- Inputs: latent, timestep, target PSNR.
- Output: one Sigmoid threshold in `[0, 1]`.
- The default 16-channel, hidden-dim-64 model has 29,249 trainable parameters.

## Validation

Ran:

```bash
/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m adaptive_threshold_predictor.inspect_trace_data
```

Observed:

- Latent shape: `(16, 12, 60, 104)`
- Timestep: `1000.0`
- Model output shape: `(1, 1)`
- Output was in `[0, 1]`

Ran:

```bash
/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m adaptive_threshold_predictor.train_gate \
  --epochs 1 \
  --batch_size 2 \
  --max_examples 8 \
  --out_dir /hy-tmp/wan22_adaptive_threshold_predictor_smoke
```

Result:

- Forward/backward training smoke test passed.
- Saved checkpoint and metrics under `/hy-tmp/wan22_adaptive_threshold_predictor_smoke`.

## Notes

- The current label builder uses existing SeaCache sweep data to create direct threshold labels, but the model and directory are named generically for the timestep-cache stage.
- No cache implementation, Wan runner, or experiment result table was modified.

## Feature Ablation Update

- Updated `ImprovedAdaCacheGate` to keep `(timestep, target_psnr)` as mandatory condition inputs.
- Added fixed-architecture feature selection through `--feature_set`.
- Supported feature sets:
  - `latent_pool`
  - `temporal_mean`
  - `temporal_var`
  - `frame_diff_mean`
  - `frame_diff_var`
- Added `run_feature_ablation.py` for batch comparison and summary writing.

Smoke command:

```bash
/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m adaptive_threshold_predictor.run_feature_ablation \
  --epochs 1 \
  --batch_size 2 \
  --max_examples 10 \
  --device cpu \
  --out_root /hy-tmp/wan22_adaptive_threshold_feature_ablation_smoke
```

Smoke result:

- All five feature sets completed.
- Each model had the same parameter count: `21057`.
- Summary saved at `/hy-tmp/wan22_adaptive_threshold_feature_ablation_smoke/feature_ablation_summary.json`.
- This run only validates the interface; it is too small for feature-importance conclusions.

## 2026-06-16 Split And Conditioning Update

- Changed train/validation split to group by `sample_id`.
- Changed timestep input from raw scheduler timestep to `step_index / (num_steps - 1)` over the 50 denoising steps.
- The model now treats timestep as already normalized to `[0, 1]` and only clamps it.
- Kept sample-level oracle labels as-is.
- Did not add high/low stage conditioning.
- Did not add optical-flow features.
- Added `feature_proj` so each selected latent-derived feature is projected to `hidden_dim` before fusion with the condition embedding.

Validation:

```bash
/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m py_compile \
  adaptive_threshold_predictor/models.py \
  adaptive_threshold_predictor/data.py \
  adaptive_threshold_predictor/train_gate.py \
  adaptive_threshold_predictor/inspect_trace_data.py \
  adaptive_threshold_predictor/run_feature_ablation.py
```

Group split smoke check:

- Built `120` examples.
- Train/val sample overlap: `0`.
- First timestep fraction: `0.0`.

Feature ablation smoke command:

```bash
/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m adaptive_threshold_predictor.run_feature_ablation \
  --epochs 1 \
  --batch_size 2 \
  --max_examples 60 \
  --device cpu \
  --out_root /hy-tmp/wan22_adaptive_threshold_feature_ablation_smoke_v2
```

Smoke result:

- All five feature sets completed.
- Each model had the same parameter count: `29377`.
- Summary saved at `/hy-tmp/wan22_adaptive_threshold_feature_ablation_smoke_v2/feature_ablation_summary.json`.
- This remains a functionality check, not a formal feature-importance result.

## 2026-06-16 Full-Step Default And PSNR Normalization

- Changed direct-threshold dataset defaults from 7 sampled denoising steps to all 50 steps.
- Current direct-threshold dataset size:
  - `100 samples * 6 target_psnr values * 50 steps = 30000 examples`
- Clarified that `100 * 10 * 50 = 50000` would require a candidate/metric-prediction dataset where each threshold candidate is an input row.
- Added train CLI arguments:
  - `--psnr_min`, default `10.0`
  - `--psnr_max`, default `50.0`
- Current normalization:
  - `timestep = step_index / 49`
  - `target_psnr_norm = clamp((target_psnr - psnr_min) / (psnr_max - psnr_min), 0, 1)`

Validation:

- Default dataset now builds `30000` examples.
- Group split gives `24000` train examples and `6000` validation examples over `80/20` samples with no sample overlap.
- Smoke training passed and saved to `/hy-tmp/wan22_adaptive_threshold_train_smoke_v3`.

## 2026-06-16 Candidate-Inverse Dataset Mode

- Added `candidate_inverse` dataset mode and made it the default.
- Each row uses:
  - candidate latent from `data/seacache/step_inputs/<threshold_label>/<sample_id>/step_*.pt`
  - `step_index / 49`
  - achieved candidate `mean_psnr`
  - label threshold used by that candidate
- Default dataset now builds `50000` examples:
  - `100 samples * 10 thresholds * 50 steps`
- Group split gives `40000` train examples and `10000` validation examples over `80/20` samples with no sample overlap.
- `target_oracle` mode remains available for comparison.
- Smoke training passed and saved to `/hy-tmp/wan22_adaptive_threshold_candidate_inverse_smoke`.

## 2026-06-16 Feature Cache And First Ablation

- Attempted raw latent five-feature training. It was too slow because each epoch repeatedly opened 50,000 step `.pt` files. The incomplete raw-latent result root only contains startup config/split files:
  - `/hy-tmp/wan22_adaptive_threshold_feature_ablation_raw_latent_20260616_011031`
- Added feature cache tooling and cached training path.
- Built cache:
  - `/hy-tmp/wan22_adaptive_threshold_feature_cache_candidate_inverse_20260616_012409`
  - `50000` examples
  - five feature tensors, each `[50000, 128]`
  - cache size about `124M`
  - build elapsed `300.38s`
- Ran cached ablation:
  - `/hy-tmp/wan22_adaptive_threshold_feature_ablation_cached_20260616_012409`
  - `candidate_inverse`, 3 epochs, batch size 256, grouped sample split
  - saved configs, splits, best/final checkpoints, metrics, validation predictions, and summary tables.

Best validation-loss ranking:

| feature_set | best_epoch | best_val_loss | best_val_mae |
| --- | ---: | ---: | ---: |
| temporal_mean | 2 | 0.012259 | 0.120107 |
| latent_pool | 2 | 0.012755 | 0.116558 |
| frame_diff_mean | 3 | 0.014569 | 0.132957 |
| temporal_var | 1 | 0.014595 | 0.129695 |
| frame_diff_var | 2 | 0.014659 | 0.131198 |

Summary files:

- `/hy-tmp/wan22_adaptive_threshold_feature_ablation_cached_20260616_012409/feature_ablation_summary.csv`
- `/hy-tmp/wan22_adaptive_threshold_feature_ablation_cached_20260616_012409/feature_ablation_best_summary.csv`
