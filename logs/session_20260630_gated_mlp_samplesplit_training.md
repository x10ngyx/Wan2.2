# 2026-06-30 Gated MLP Sample-Split Training

## Summary

- Ran the normal sample-level train/validation split for the gated 4-feature MLP.
- Training completed before the GPU was turned off.
- No inference/video generation was run.

## Result Root

```text
/hy-tmp/wan22_adaptive_threshold_mlp_gated_4feature_samplesplit_20260630_013006
```

Workspace symlink:

```text
experiment_results/wan22_adaptive_threshold_mlp_gated_4feature_samplesplit_20260630_013006
```

Launch/log:

```text
/hy-tmp/wan22_adaptive_threshold_mlp_gated_4feature_samplesplit_20260630_013006/commands/launch_train.sh
/hy-tmp/wan22_adaptive_threshold_mlp_gated_4feature_samplesplit_20260630_013006/logs/train.log
```

## Command Shape

```bash
/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m adaptive_threshold_predictor.train_gate \
  --model_type mlp \
  --cache_dir /hy-tmp/wan22_adaptive_threshold_feature_cache_candidate_inverse_20260616_012409 \
  --feature_sets latent_pool temporal_var frame_diff_mean frame_diff_var \
  --dataset_mode candidate_inverse \
  --split_mode sample \
  --epochs 30 \
  --batch_size 256 \
  --hidden_dim 64 \
  --feature_embedding_dim 64 \
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
  --save_val_predictions
```

## Results

- Train examples: `40000`
- Validation examples: `10000`
- Train samples: `80`
- Validation samples: `20`
- Parameters: `71045`
- Epochs run: `13`
- Early stopped: epoch `13`
- Best epoch: `8`
- Best validation MAE: `0.11151796581298112`
- Best validation loss: `0.10222012972831726`
- Best epoch train MAE: `0.08612967762444168`
- Final epoch validation MAE: `0.11888088526204228`

Best epoch validation MAE by step:

```text
step_00_09: 0.11781573643535376
step_10_39: 0.10946383323520421
step_40_49: 0.11138259292393923
```

Best epoch validation MAE by threshold:

```text
0.10: 0.015568429000675678
0.15: 0.03470603171736002
0.20: 0.05534320007264614
0.25: 0.09033234791457653
0.30: 0.1060278910547495
0.40: 0.11715792307257653
0.50: 0.12313540376722813
0.60: 0.12578511033952236
0.70: 0.18056872788071632
0.80: 0.2665545933097601
```

Final checkpoint validation gate means:

```text
gate_latent_pool: 0.6956643772833049
gate_temporal_var: 0.09785542918057182
gate_frame_diff_mean: 0.1012345067290822
gate_frame_diff_var: 0.10524568697217619
```

## Interpretation

- Sample-split gated MLP reached best val MAE `0.1115`, close to the previous sample-split MiniDiT best `0.1144591414630413`, while using many fewer parameters.
- High-threshold labels remain the hardest, especially `0.70` and `0.80`.
- Final checkpoint gate means are dominated by `latent_pool`; motion-derived features still receive nonzero weight.
- Gate-by-threshold inspection shows motion features are weighted more at low threshold `0.10` and less at high thresholds.

## Notes

- User turned off GPU after training completed; no further GPU work was needed.
- No git commit was made.
