# Gated Multi-Feature MLP Threshold Predictor Architecture Proposal

Date: 2026-06-29

![Gated multi-feature MLP architecture](assets/gated_multifeature_mlp_architecture.svg)

## 1. Objective

This document defines the gated multi-feature MLP adaptive threshold predictor
for Wan2.2 T2V SeaCache-style inference acceleration.

The goal is to extend the original pooled-feature MLP without moving all the way
to a Transformer. The model should answer a focused question:

```text
Does combining several latent-derived summary features improve threshold
prediction when each feature is encoded separately and fused by a learned gate?
```

The model deliberately does not use direct feature concatenation. Multiple
features are treated as separate signals:

- each feature has its own MLP encoder,
- timestep and target PSNR produce a condition embedding,
- the condition embedding predicts a softmax gate over feature encoders,
- the gated weighted feature embedding is fused with the condition embedding,
- the output is one scalar threshold.

This keeps the architecture small, interpretable, and comparable to the earlier
single-feature MLP ablations.

## 2. Data Constraint

The current default dataset mode is still `candidate_inverse`:

```text
input: candidate latent-derived feature at current denoising step,
       step_index / 49,
       achieved PSNR
label: threshold used by the candidate run
```

With OpenVid-100 SeaCache traces this gives:

```text
100 samples * 10 threshold candidates * 50 denoising steps = 50,000 examples
```

The usual sample-level train/validation split gives about 40k training examples
and 10k validation examples. These rows are strongly correlated because they
come from a much smaller number of prompt/video trajectories. The predictor
therefore remains small: about 84k parameters for the recommended first run.

## 3. Input Features

The model uses five pooled latent-derived features selected from the earlier MLP
feature ablations:

```text
latent_pool
temporal_mean
temporal_var
frame_diff_mean
frame_diff_var
```

Feature definitions:

| Feature | Meaning |
|---|---|
| `latent_pool` | raw latent pooled to a compact 3D grid |
| `temporal_mean` | per-channel temporal mean, then pooled |
| `temporal_var` | per-channel temporal variance, then pooled |
| `frame_diff_mean` | first-order absolute frame-difference mean, then pooled |
| `frame_diff_var` | first-order absolute frame-difference variance, then pooled |

The gated run now includes `temporal_mean` as a fifth feature. The selected
features cover raw latent magnitude, temporal average content, temporal
instability, and frame-to-frame change. Each feature remains encoded separately
and fused only after the learned gate.

For the current pooled feature cache, each feature has:

```text
latent_channels = 16
pool grid = [2, 2, 2]
feature_dim = 16 * 2 * 2 * 2 = 128
```

Cached feature files:

```text
features_latent_pool.pt
features_temporal_mean.pt
features_temporal_var.pt
features_frame_diff_mean.pt
features_frame_diff_var.pt
```

The cached dataset returns per-feature tensors, not a concatenated tensor:

```text
batch["features"]["latent_pool"]       -> [B, 128]
batch["features"]["temporal_mean"]     -> [B, 128]
batch["features"]["temporal_var"]      -> [B, 128]
batch["features"]["frame_diff_mean"]   -> [B, 128]
batch["features"]["frame_diff_var"]    -> [B, 128]
```

For raw-latent training without `--cache_dir`, `GatedMultiFeatureAdaCacheGate`
extracts the same five feature tensors internally from:

```text
latent [B, 16, 12, 60, 104]
```

## 4. Proposed Architecture

Recommended model path:

```text
CachedGatedFeatureAdaCacheGate
```

Raw-latent equivalent:

```text
GatedMultiFeatureAdaCacheGate
```

High-level flow:

```text
feature_i [B, 128]
  -> feature_encoder_i MLP
  -> feature_embedding_i [B, 64]

condition [step_fraction, target_psnr_norm]
  -> condition MLP
  -> condition_embedding [B, 64]

condition_embedding
  -> gate MLP
  -> softmax gate [B, 5]

fused_feature = sum_i gate_i * feature_embedding_i

concat(fused_feature, condition_embedding)
  -> prediction head
  -> threshold [B, 1]
```

No direct multi-feature concatenation is used. Concatenation appears only after
gated fusion, where the fused feature embedding is combined with the condition
embedding for the final prediction head.

## 5. Hyperparameters

Recommended first-run hyperparameters:

| Item | Value |
|---|---:|
| feature set | `latent_pool temporal_mean temporal_var frame_diff_mean frame_diff_var` |
| per-feature input dim | `128` |
| number of features | `5` |
| hidden dim | `64` |
| feature embedding dim | `64` |
| condition input dim | `2` |
| condition embedding dim | `64` |
| feature encoder depth | `2 Linear layers` |
| gate type | condition-dependent softmax |
| gate output dim | `5` |
| activation | `SiLU` |
| dropout | `0.05` for first full run |
| output activation | scaled `Sigmoid` over a raw logit |
| output range | `[0.10, 0.80]` by default |

The gated MLP now uses the same threshold range mapping as the MiniDiT/Transformer
predictor:

```text
threshold = min_threshold + sigmoid(raw) * (max_threshold - min_threshold)
```

With the current defaults, `min_threshold=0.10` and `max_threshold=0.80`. This
matches the SeaCache candidate label range and avoids the previous semantic
mismatch where the MLP head could emit values in `[0, 1]`.

## 6. Module Details

Per-feature encoder for each feature:

```text
Linear(128 -> 64)
SiLU
Dropout(0.05)
Linear(64 -> 64)
SiLU
```

Condition input:

```text
step_fraction = step_index / (num_steps - 1)
target_psnr_norm = clamp((target_psnr - 10.0) / 40.0, 0, 1)
```

Condition encoder:

```text
Linear(2 -> 64)
SiLU
Linear(64 -> 64)
SiLU
```

Gate head:

```text
Linear(64 -> 64)
SiLU
Linear(64 -> 5)
Softmax(dim=-1)
```

Fusion:

```text
encoded = stack([z_latent_pool, z_temporal_mean, z_temporal_var, z_frame_diff_mean, z_frame_diff_var])
gate = softmax(gate_head(condition_embedding))
fused_feature = sum(encoded * gate[..., None], dim=feature_axis)
```

Prediction head:

```text
concat(fused_feature, condition_embedding)  # [B, 128]
Linear(128 -> 64)
LayerNorm(64)
SiLU
Dropout(0.05)
Linear(64 -> 64)
SiLU
Linear(64 -> 1)
raw threshold logit
scaled Sigmoid to [min_threshold, max_threshold]
```

## 7. Parameter Count

Measured with the current implementation and recommended first-run settings:

```text
hidden_dim = 64
feature_embedding_dim = 64
feature_dim per feature = 128
num_features = 5
```

Total trainable parameters:

```text
83,526
```

Breakdown:

| Module | Parameters |
|---|---:|
| feature encoders total | `62,080` |
| each feature encoder | `12,416` |
| condition encoder | `4,352` |
| gate head | `4,485` |
| prediction head | `12,609` |
| total | `83,526` |

Formula check:

```text
feature encoder per feature:
  Linear(128 -> 64): 128*64 + 64 = 8,256
  Linear(64 -> 64):   64*64 + 64 = 4,160
  total per feature: 12,416

5 feature encoders:
  12,416 * 5 = 62,080

condition encoder:
  Linear(2 -> 64):   2*64 + 64 = 192
  Linear(64 -> 64): 64*64 + 64 = 4,160
  total: 4,352

gate head:
  Linear(64 -> 64): 64*64 + 64 = 4,160
  Linear(64 -> 5):  64*5 + 5 = 325
  total: 4,485

prediction head:
  Linear(128 -> 64): 128*64 + 64 = 8,256
  LayerNorm(64): 64 weight + 64 bias = 128
  Linear(64 -> 64): 64*64 + 64 = 4,160
  Linear(64 -> 1): 64*1 + 1 = 65
  total: 12,609
```

The cached-feature and raw-latent gated models have the same trainable parameter
count. Raw-latent feature extraction uses fixed pooling/statistics and does not
add learnable parameters.

## 8. Training Command

Recommended first full run:

```bash
/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m adaptive_threshold_predictor.train_gate \
  --model_type mlp \
  --cache_dir /hy-tmp/wan22_adaptive_threshold_feature_cache_candidate_inverse_20260616_012409 \
  --feature_sets latent_pool temporal_mean temporal_var frame_diff_mean frame_diff_var \
  --dataset_mode candidate_inverse \
  --epochs 30 \
  --batch_size 256 \
  --hidden_dim 64 \
  --feature_embedding_dim 64 \
  --min_threshold 0.10 \
  --max_threshold 0.80 \
  --lr 3e-4 \
  --min_lr 1e-5 \
  --warmup_steps 500 \
  --weight_decay 1e-4 \
  --smooth_l1_beta 0.02 \
  --grad_clip 1.0 \
  --early_stop_patience 5 \
  --dit_dropout 0.05 \
  --device cuda \
  --num_workers 4 \
  --save_val_predictions \
  --out_dir /hy-tmp/wan22_adaptive_threshold_mlp_gated_5feature
```

Notes:

- `--feature_sets` with more than one feature selects the gated multi-feature
  path.
- `--feature_set` remains available for single-feature legacy MLP ablations.
- There is no multi-feature direct-concat mode.

## 9. Interpretability Output

When `--save_val_predictions` is enabled, validation predictions include gate
weights:

```text
gate_latent_pool
gate_temporal_mean
gate_temporal_var
gate_frame_diff_mean
gate_frame_diff_var
```

These columns make it possible to inspect feature usage by:

- denoising step range,
- target PSNR bucket,
- label threshold bucket,
- sample/prompt.

Useful diagnostics:

```text
mean gate by step bucket
mean gate by target PSNR
mean gate by label threshold
gate entropy by epoch
correlation between gate choice and absolute threshold error
```

If the gate collapses to one feature for all examples, the multi-feature model
is acting like a larger single-feature MLP. If the gate changes systematically
with step or target PSNR, the model is using the intended conditional fusion
mechanism.

## 10. Expected Comparison

The first comparison should be:

| Model | Features | Fusion | Purpose |
|---|---|---|---|
| single-feature MLP | `latent_pool` | none | old baseline |
| single-feature MLP | `temporal_var` | none | best/competitive old feature candidate |
| single-feature MLP | `frame_diff_mean` | none | motion-focused single feature |
| single-feature MLP | `frame_diff_var` | none | motion-variance single feature |
| gated MLP | five selected features | softmax gate | proposed multi-feature model |
| MiniDiT-CLS | raw latent tokens | attention | higher-capacity comparator |

Primary offline metrics:

- validation SmoothL1 loss,
- validation MAE on threshold,
- prediction range and bias,
- bucketed MAE by step, target PSNR, and threshold label,
- gate-weight summaries.

The real test remains adaptive inference quality/speed, because `candidate_inverse`
training has a known mismatch with online adaptive inference. Offline threshold
MAE alone should not be treated as final evidence.

## 11. Implementation Status

Implemented files:

- `adaptive_threshold_predictor/models.py`
- `adaptive_threshold_predictor/data.py`
- `adaptive_threshold_predictor/train_gate.py`

Implemented model classes:

- `GatedFeatureFusionAdaCacheGate`
- `CachedGatedFeatureAdaCacheGate`
- `GatedMultiFeatureAdaCacheGate`

Validation completed:

- syntax check with `py_compile`,
- cached gated CPU smoke,
- cached gated CPU smoke with non-empty validation and gate columns,
- raw-latent random forward smoke,
- parameter count measurement.

Full 5-feature GPU retraining completed on 2026-06-30 after adding
`temporal_mean` to the gated feature set:

| Run | Split | Epochs Run | Best Epoch | Best Val MAE | Best Val Loss |
|---|---|---:|---:|---:|---:|
| `wan22_adaptive_threshold_mlp_gated_5feature_samplesplit_20260630_021641` | sample | 12 / 30, early stopped | 7 | `0.1142528785` | `0.1050056725` |
| `wan22_adaptive_threshold_mlp_gated_5feature_rowsplit_gpu_20260630_021641` | row | 30 / 30 | 30 | `0.0756697811` | `0.0667008773` |
| `wan22_adaptive_threshold_mlp_gated_5feature_rowsplit_gpu_long100_20260630_021641` | row | 100 / 100 | 98 | `0.0601118673` | `0.0513865515` |

Mean validation gate weights:

| Run | latent_pool | temporal_mean | temporal_var | frame_diff_mean | frame_diff_var |
|---|---:|---:|---:|---:|---:|
| sample split 30 | `0.5146` | `0.2312` | `0.0784` | `0.0689` | `0.1068` |
| row split 30 | `0.4112` | `0.2113` | `0.1137` | `0.1136` | `0.1502` |
| row split 100 | `0.3906` | `0.1825` | `0.1439` | `0.1116` | `0.1714` |

After fixing the output-range mismatch, the same three 5-feature runs were
retrained with the scaled-sigmoid threshold head:

| Run | Split | Epochs Run | Best Epoch | Best Val MAE | Best Val Loss | Best Prediction Range |
|---|---|---:|---:|---:|---:|---|
| `wan22_adaptive_threshold_mlp_gated_5feature_range_samplesplit_20260630_035000` | sample | 14 / 30, early stopped | 9 | `0.1143567288` | `0.1049280047` | `[0.1023, 0.7303]` |
| `wan22_adaptive_threshold_mlp_gated_5feature_range_rowsplit_gpu_20260630_035000` | row | 30 / 30 | 30 | `0.0770497653` | `0.0678418150` | `[0.1040, 0.7743]` |
| `wan22_adaptive_threshold_mlp_gated_5feature_range_rowsplit_gpu_long100_20260630_035000` | row | 100 / 100 | 98 | `0.0610311001` | `0.0522424302` | `[0.1010, 0.7985]` |

Comparison against the previous direct-`Sigmoid` 5-feature MLP:

| Split / Budget | Previous `[0,1]` MAE | Range-Mapped `[0.1,0.8]` MAE | Delta |
|---|---:|---:|---:|
| sample split 30 | `0.1142528785` | `0.1143567288` | `+0.0001038503` |
| row split 30 | `0.0756697811` | `0.0770497653` | `+0.0013799842` |
| row split 100 | `0.0601118673` | `0.0610311001` | `+0.0009192327` |

Mean validation gate weights for the range-mapped runs:

| Run | latent_pool | temporal_mean | temporal_var | frame_diff_mean | frame_diff_var |
|---|---:|---:|---:|---:|---:|
| sample split 30 | `0.5464` | `0.1839` | `0.0945` | `0.0986` | `0.0766` |
| row split 30 | `0.4781` | `0.1967` | `0.1267` | `0.0998` | `0.0987` |
| row split 100 | `0.3987` | `0.2533` | `0.0942` | `0.0962` | `0.1576` |

Interpretation: the range mapping fixes the online-semantics mismatch and all
saved predictions now stay inside `[0.10, 0.80]`. It does not improve offline
threshold MAE on this candidate-inverse dataset; the changes are essentially
neutral on sample split and slightly worse on row split. The 100-epoch row split
still improves substantially over 30 epochs, but remains behind the MiniDiT
row-split reference MAE `0.0380019387`.
