# Adaptive Threshold Predictor

This top-level package contains the adaptive threshold prediction work, separate
from the Wan2.2 generation runners. The current stage is timestep-cache-only, so
`ImprovedAdaCacheGate` predicts one SeaCache threshold.

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
- prediction head: fixed small MLP producing one raw logit

Output:

```text
threshold: [B, 1], value range [min_threshold, max_threshold]
```

The MLP family uses the same output mapping as the MiniDiT/Transformer
predictor:

```text
threshold = min_threshold + sigmoid(raw) * (max_threshold - min_threshold)
```

The default range is `[0.10, 0.80]`, matching the SeaCache threshold candidates
used by the current cached-feature training data.

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
/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m adaptive_threshold_predictor.inspect_trace_data
```

Run a small debug training loop:

```bash
/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m adaptive_threshold_predictor.train_gate \
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
/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m adaptive_threshold_predictor.run_feature_ablation \
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
/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m adaptive_threshold_predictor.build_feature_cache \
  --out_dir /hy-tmp/wan22_adaptive_threshold_feature_cache_candidate_inverse_20260616_012409 \
  --dataset_mode candidate_inverse \
  --dtype float32 \
  --batch_size 8 \
  --num_workers 4 \
  --device cuda
```

Then train from the cache:

```bash
/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m adaptive_threshold_predictor.run_feature_ablation \
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

## Multi-Feature MLP

The legacy single-feature MLP path remains available through `--feature_set` for
the original ablations. Multi-feature MLP runs use gated fusion: each feature is
encoded by its own small MLP, and the timestep/PSNR condition branch predicts a
softmax gate over feature embeddings.

```bash
/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m adaptive_threshold_predictor.train_gate \
  --model_type mlp \
  --cache_dir /hy-tmp/wan22_adaptive_threshold_feature_cache_candidate_inverse_20260616_012409 \
  --feature_sets latent_pool temporal_mean temporal_var frame_diff_mean frame_diff_var \
  --dataset_mode candidate_inverse \
  --epochs 30 \
  --batch_size 256 \
  --hidden_dim 64 \
  --min_threshold 0.10 \
  --max_threshold 0.80 \
  --device cuda \
  --num_workers 4 \
  --out_dir /hy-tmp/wan22_adaptive_threshold_mlp_gated_5feature
```

With `--cache_dir`, the dataset loads each corresponding
`features_<feature_set>.pt` file and returns per-feature tensors to the gated
model. Without `--cache_dir`, `GatedMultiFeatureAdaCacheGate` extracts the same
feature set list from raw latents at training time.

The recommended first gated feature set is:

```text
latent_pool temporal_mean temporal_var frame_diff_mean frame_diff_var
```

This uses the full cached pooled-feature set: raw pooled latent, temporal mean,
temporal variance, and first-order frame-difference mean/variance. The
multi-feature path still keeps these as separate per-feature tensors and fuses
them with a learned softmax gate; it does not concatenate the raw feature
vectors directly.

## MiniDiT CLS Predictor

The recommended MiniDiT-CLS model uses raw traced latents and a learned Conv3d
patch embedding:

```text
latent [B,16,12,60,104]
  -> Conv3d(16, 96, kernel_size=(3,12,8), stride=(3,12,8))
  -> tokens over grid [4,5,13]
  -> CLS Transformer readout
```

Train it directly from the trace data:

```bash
/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m adaptive_threshold_predictor.train_gate \
  --model_type mini_dit_cls \
  --out_dir /hy-tmp/wan22_adaptive_threshold_mini_dit_cls_3x12x8_d96_l2 \
  --dataset_mode candidate_inverse \
  --batch_size 64 \
  --epochs 30 \
  --lr 3e-4 \
  --min_lr 1e-5 \
  --warmup_steps 500 \
  --weight_decay 1e-4 \
  --smooth_l1_beta 0.02 \
  --grad_clip 1.0 \
  --early_stop_patience 5 \
  --dit_dim 96 \
  --dit_layers 2 \
  --dit_heads 4 \
  --dit_mlp_ratio 2.0 \
  --dit_dropout 0.05 \
  --dit_patch_size 3 12 8 \
  --min_threshold 0.10 \
  --max_threshold 0.80 \
  --save_val_predictions \
  --save_epoch_val_predictions \
  --device cuda \
  --num_workers 4
```

The checkpoint metadata includes the feature extractor configuration:

```text
type = learned_conv3d_patch_embedding
input_shape = [16, 12, 60, 104]
patch_size = [3, 12, 8]
token_grid_shape = [4, 5, 13]
token_count = 260
```

The MiniDiT training path writes:

```text
config.json
split.json
model_summary.json
epoch_metrics.jsonl
epoch_metrics.csv
metrics.json
best_model.pt
best_model_checkpoint.pt
final_model.pt
final_model_checkpoint.pt
val_predictions.csv
val_predictions_epoch_*.csv  # only with --save_epoch_val_predictions
```
