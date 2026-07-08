# Adaptive Threshold Predictor

This top-level package contains the adaptive threshold prediction work, separate
from the Wan2.2 generation runners. The current stage is timestep-cache-only, so
`ImprovedAdaCacheGate` predicts one SeaCache threshold.

## File Map

Use `train_gate.py` as the main training entry point. The other scripts either
build faster training caches or launch ablation batches.

```text
README.md                   This handoff and usage guide.
__init__.py                 Package marker only.
data.py                     Dataset builders, cached dataset readers, collate
                            functions, and sample/row split helpers.
models.py                   MLP, gated multi-feature MLP, condition-only,
                            grid-MLP, and MiniDiT-CLS threshold predictors.
train_gate.py               Main trainer. Writes config, split, metrics,
                            checkpoints, and validation predictions.
inspect_trace_data.py       Small smoke check for one traced latent and one
                            untrained model forward pass.
build_feature_cache.py      Builds pooled 2x2x2-style feature caches for MLP
                            and gated MLP training.
build_grid_feature_cache.py Builds fixed avg-pool 3D grid caches for the
                            grid_mlp capacity baseline.
build_raw_latent_cache.py   Packs raw step latents into shard files for faster
                            MiniDiT-CLS training.
run_feature_ablation.py     Runs train_gate.py once per feature set and writes
                            a feature-ablation summary table.
run_grid_ablation.py        Builds pooled-feature caches for several grid sizes
                            and runs run_feature_ablation.py on each.
```

The package assumes the repo root is on `PYTHONPATH`, which is true when running
commands from `/hy-tmp/work/Wan2.2` with `python -m adaptive_threshold_predictor...`.

## Data

Default trace data root:

```bash
/hy-tmp/openvid_100_seacache_trace_data
```

The summary CSV is read from:

```text
/hy-tmp/openvid_100_seacache_trace_data/data/tables/summary.csv
```

Observed single-step latent tensors are saved as:

```text
[C, T, H, W] = [16, 12, 60, 104]
```

The model accepts both single trace tensors `[C, T, H, W]` and batched tensors
`[B, C, T, H, W]`.

## Model

`ImprovedAdaCacheGate` uses:

- condition branch: lightweight MLP over
  `(step_index / 49, target_psnr, target_speedup)`,
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

Current training uses `candidate_inverse` labels. Each measured SeaCache
candidate contributes one example per denoising step:

```text
input: candidate latent at current step, step_index / 49, achieved PSNR,
       achieved speedup
label: threshold used by the candidate run
```

In the code these achieved values are named `target_psnr` and `target_speedup`
because the online predictor receives desired target values at inference time.
For offline inverse training they are measured candidate outcomes, not manually
configured targets.

With the current data this gives:

```text
100 samples * 10 threshold candidates * 50 steps = 50000 examples
```

The condition branch normalizes PSNR and speedup as:

```text
psnr_norm = clamp((target_psnr - psnr_min) / (psnr_max - psnr_min), 0, 1)
speedup_norm = clamp((target_speedup - speedup_min) / (speedup_max - speedup_min), 0, 1)
```

Defaults:

```text
psnr_min = 10
psnr_max = 50
speedup_min = 1
speedup_max = 4
```

These bounds cover the observed table ranges while leaving margin around the
online target PSNR and speedup values used by adaptive inference experiments.

## Quick Checks

Inspect a traced latent and run one model forward pass:

```bash
/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m adaptive_threshold_predictor.inspect_trace_data
```

Run a small debug training loop:

```bash
/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m adaptive_threshold_predictor.train_gate \
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

For quick checks, keep `--max_examples` small if the full OpenVid trace root is
not present or if you only need to validate imports and tensor shapes.

## Cached Features

Raw latent training repeatedly opens 50,000 step `.pt` files and is too slow for
multi-run ablations. Build pooled feature caches once:

```bash
/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m adaptive_threshold_predictor.build_feature_cache \
  --out_dir /hy-tmp/wan22_adaptive_threshold_feature_cache_candidate_inverse_20260616_012409 \
  --dtype float32 \
  --batch_size 8 \
  --num_workers 4 \
  --device cuda
```

Then train from the cache:

```bash
/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m adaptive_threshold_predictor.run_feature_ablation \
  --cache_dir /hy-tmp/wan22_adaptive_threshold_feature_cache_candidate_inverse_20260616_012409 \
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
encoded by its own small MLP, and the timestep/PSNR/speedup condition branch
predicts a softmax gate over feature embeddings.

```bash
/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m adaptive_threshold_predictor.train_gate \
  --model_type mlp \
  --cache_dir /hy-tmp/wan22_adaptive_threshold_feature_cache_candidate_inverse_20260616_012409 \
  --feature_sets latent_pool temporal_mean temporal_var frame_diff_mean frame_diff_var \
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

For repeated MiniDiT training, first pack raw latents into shards so training
does not reopen 50,000 individual step files:

```bash
/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m adaptive_threshold_predictor.build_raw_latent_cache \
  --out_dir /hy-tmp/wan22_adaptive_threshold_raw_latent_cache_fp16 \
  --dtype float16 \
  --shard_size 512 \
  --batch_size 16 \
  --num_workers 2
```

Train from the packed cache:

```bash
/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m adaptive_threshold_predictor.train_gate \
  --model_type mini_dit_cls \
  --packed_latent_cache_dir /hy-tmp/wan22_adaptive_threshold_raw_latent_cache_fp16 \
  --out_dir /hy-tmp/wan22_adaptive_threshold_mini_dit_cls_3x12x8_d96_l2 \
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

You can omit `--packed_latent_cache_dir` to train directly from individual trace
step files, but that path is slower and mainly useful for small smoke runs.

## Grid MLP Baseline

`grid_mlp` is a capacity baseline over fixed avg-pooled 3D grid features. It is
separate from MiniDiT: MiniDiT learns its Conv3d patch embedding from raw
latents, while `grid_mlp` consumes a prebuilt `grid_features.pt` cache.

```bash
/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m adaptive_threshold_predictor.build_grid_feature_cache \
  --out_dir /hy-tmp/wan22_adaptive_threshold_grid_cache_3x12x8 \
  --patch_size 3 12 8 \
  --dtype float16 \
  --batch_size 8 \
  --num_workers 4 \
  --device cuda

/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m adaptive_threshold_predictor.train_gate \
  --model_type grid_mlp \
  --grid_cache_dir /hy-tmp/wan22_adaptive_threshold_grid_cache_3x12x8 \
  --out_dir /hy-tmp/wan22_adaptive_threshold_grid_mlp_3x12x8 \
  --batch_size 256 \
  --epochs 30 \
  --device cuda
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

## Split Modes

Use `--split_mode sample` for the primary held-out-video signal. It keeps all
rows from the same `sample_id` on one side of the split.

Use `--split_mode row` only as a same-video interpolation diagnostic. Row split
can share a source video between train and validation, so low row-split MAE is
not evidence of held-out-video generalization.

## Current Caveats

- The predictor outputs one timestep-cache SeaCache threshold only. It does not
  predict block-cache or CFG-cache thresholds.
- Existing two-condition checkpoints from before `target_speedup` was added are
  incompatible with the current three-condition model constructors.
- Offline `candidate_inverse` MAE is not the same as online target-control
  quality. Final claims still need online adaptive SeaCache validation against
  fixed-threshold controls on the same prompts.
- Large trace data, feature caches, raw-latent shards, and checkpoints should
  stay under `/hy-tmp`, not inside the git repository.
