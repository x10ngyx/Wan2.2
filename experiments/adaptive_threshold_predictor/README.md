# Adaptive Threshold Predictor

This subdirectory keeps the prediction-network work separate from the Wan2.2
generation runners. The current stage is timestep-cache-only, so
`ImprovedAdaCacheGate` predicts one threshold in `[0, 1]`.

## Data

Default trace data root:

```bash
/hy-tmp/openvid_100_seacache_trace_data/data
```

Observed single-step latent tensors are saved as:

```text
[C, T, H, W] = [16, 12, 60, 104]
```

The model accepts both single trace tensors `[C, T, H, W]` and batched tensors
`[B, C, T, H, W]`.

## Model

`ImprovedAdaCacheGate` uses:

- condition branch: lightweight MLP over `(step_index / 49, target_psnr)`,
  always enabled for the 50-step traces
- feature branch: one selectable latent-derived feature, projected with the same
  `AdaptiveAvgPool3d((2, 2, 2))` output shape
- feature projector: maps the pooled feature to `hidden_dim` before fusion, so
  the prediction head stays fixed when testing different feature inputs
- prediction head: fixed small MLP with a final `Sigmoid`

Output:

```text
threshold: [B, 1], value range [0, 1]
```

The feature ablation keeps the condition branch and prediction head fixed, and
only changes the latent-derived feature input:

```text
latent_pool       raw latent pooled to 2x2x2
temporal_mean     temporal mean of latent, then pooled to 2x2x2
temporal_var      temporal variance of latent, then pooled to 2x2x2
frame_diff_mean   first-order absolute frame-difference mean, then pooled
frame_diff_var    first-order absolute frame-difference variance, then pooled
```

This setup is intended to compare validation loss across feature factors while
minimizing architecture differences.

The train/validation split is grouped by `sample_id`: all target PSNRs and
sampled denoising steps from the same source sample stay on the same side of the
split.

The default dataset mode is `candidate_inverse`. Each measured SeaCache
candidate contributes one example per denoising step:

```text
input: candidate latent at current step, step_index / 49, achieved PSNR
label: threshold used by the candidate run
```

With the current data this gives:

```text
100 samples * 10 threshold candidates * 50 steps = 50000 examples
```

The older `target_oracle` mode is still available for comparison:

```text
input: baseline latent at current step, step_index / 49, desired target PSNR
label: fastest measured threshold satisfying that target PSNR for the sample
```

The condition branch normalizes PSNR as:

```text
psnr_norm = clamp((target_psnr - psnr_min) / (psnr_max - psnr_min), 0, 1)
```

Defaults:

```text
psnr_min = 10
psnr_max = 50
```

These bounds cover the observed table range while leaving margin around the
configured target PSNR values.

## Quick Checks

Inspect a traced latent and run one model forward pass:

```bash
/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m experiments.adaptive_threshold_predictor.inspect_trace_data
```

Run a small debug training loop:

```bash
/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m experiments.adaptive_threshold_predictor.train_gate \
  --dataset_mode candidate_inverse \
  --epochs 3 \
  --batch_size 4 \
  --feature_set temporal_var \
  --psnr_min 10 \
  --psnr_max 50 \
  --out_dir /hy-tmp/wan22_adaptive_threshold_predictor_debug
```

Run all feature-set ablations with the same architecture and collect a summary:

```bash
/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m experiments.adaptive_threshold_predictor.run_feature_ablation \
  --epochs 3 \
  --batch_size 4 \
  --out_root /hy-tmp/wan22_adaptive_threshold_feature_ablation
```

The first label builder constructs direct-threshold labels from the SeaCache
sweep table: for each sample and target PSNR, it selects the fastest threshold
whose measured PSNR reaches the target, or the highest-PSNR threshold if the
target is unreachable.

## Cached Features

Raw latent training repeatedly opens 50,000 step `.pt` files and is too slow for
multi-run ablations. Build pooled feature caches once:

```bash
/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m experiments.adaptive_threshold_predictor.build_feature_cache \
  --out_dir /hy-tmp/wan22_adaptive_threshold_feature_cache_candidate_inverse_20260616_012409 \
  --dataset_mode candidate_inverse \
  --dtype float32 \
  --batch_size 8 \
  --num_workers 4 \
  --device cuda
```

Then train from the cache:

```bash
/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m experiments.adaptive_threshold_predictor.run_feature_ablation \
  --cache_dir /hy-tmp/wan22_adaptive_threshold_feature_cache_candidate_inverse_20260616_012409 \
  --dataset_mode candidate_inverse \
  --epochs 3 \
  --batch_size 256 \
  --device cuda \
  --save_val_predictions \
  --out_root /hy-tmp/wan22_adaptive_threshold_feature_ablation_cached_20260616_012409
```

The cached run saves per-feature configs, splits, best/final checkpoints,
metrics, validation predictions, and summary CSV/JSON files.
