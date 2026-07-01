# 2026-06-30 Gated MLP Row-Split Training

## Summary

- Started a CPU row-split gated MLP training attempt while GPU was unavailable.
- CPU training was too slow and produced no epoch output after several minutes.
- After GPU was restored, removed the CPU attempt residual and ran the row-split training on GPU.
- Training completed successfully.

## Cleanup

Removed CPU attempt residual:

```text
/hy-tmp/wan22_adaptive_threshold_mlp_gated_4feature_rowsplit_cpu_20260630_014051
experiment_results/wan22_adaptive_threshold_mlp_gated_4feature_rowsplit_cpu_20260630_014051
```

## Result Root

```text
/hy-tmp/wan22_adaptive_threshold_mlp_gated_4feature_rowsplit_gpu_20260630_014852
```

Workspace symlink:

```text
experiment_results/wan22_adaptive_threshold_mlp_gated_4feature_rowsplit_gpu_20260630_014852
```

Launch/log:

```text
/hy-tmp/wan22_adaptive_threshold_mlp_gated_4feature_rowsplit_gpu_20260630_014852/commands/launch_train.sh
/hy-tmp/wan22_adaptive_threshold_mlp_gated_4feature_rowsplit_gpu_20260630_014852/logs/train.log
```

## Command Shape

```bash
/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m adaptive_threshold_predictor.train_gate \
  --model_type mlp \
  --cache_dir /hy-tmp/wan22_adaptive_threshold_feature_cache_candidate_inverse_20260616_012409 \
  --feature_sets latent_pool temporal_var frame_diff_mean frame_diff_var \
  --dataset_mode candidate_inverse \
  --split_mode row \
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
- Split mode: `row`
- Parameters: `71045`
- Epochs run: `30`
- Early stopped: no
- Best epoch: `30`
- Best validation MAE: `0.07593761396706104`
- Best validation loss: `0.06691185193061829`
- Best epoch train MAE: `0.07653351860381663`

Best epoch validation MAE by step:

```text
step_00_09: 0.0928498020344101
step_10_39: 0.07261506491243311
step_40_49: 0.0687483228470994
```

Best epoch validation MAE by threshold:

```text
0.10: 0.014997302341942836
0.15: 0.03539148792775295
0.20: 0.045847805324606423
0.25: 0.06859489056477255
0.30: 0.07987874489255815
0.40: 0.10076054226516223
0.50: 0.0864502436272883
0.60: 0.07383361327600999
0.70: 0.09040507235947777
0.80: 0.16622635118142076
```

Final checkpoint validation gate means:

```text
gate_latent_pool: 0.602346860973537
gate_temporal_var: 0.1366311730541289
gate_frame_diff_mean: 0.14120697866557166
gate_frame_diff_var: 0.11981498754592612
```

## Interpretation

- Row split improved over the sample-split gated MLP (`0.07594` vs `0.11152` best val MAE).
- Row-split gated MLP remains worse than row-split MiniDiT (`0.03800` best val MAE), but uses many fewer parameters.
- Gate weights are less dominated by `latent_pool` than in sample split, with more weight on temporal and frame-difference features.
- Threshold `0.80` remains the hardest bucket.
- GPU was idle again after completion.

## Notes

- No video inference or PSNR evaluation was run.
- No git commit was made.
