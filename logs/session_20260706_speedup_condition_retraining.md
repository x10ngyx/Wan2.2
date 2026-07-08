# 2026-07-06 Speedup-Conditioned Predictor Retraining

## Scope

- Continued the `target_speedup` predictor-input work by checking the completed retraining runs.
- No training code was changed in this step.
- No git commit was made.

## Training Launch

- Aggregate root: `/hy-tmp/wan22_speedup_condition_retrain_20260706_171523`
- Aggregate script: `/hy-tmp/wan22_speedup_condition_retrain_20260706_171523/commands/run_all.sh`
- Aggregate log: `/hy-tmp/wan22_speedup_condition_retrain_20260706_171523/logs/run_all.log`
- tmux session `wan22_speedup_condition_retrain_20260706_171523` completed and exited.

## Completed Runs

| Model | Split | Root | Epochs | Best epoch | Best val MAE | Early stop |
| --- | --- | --- | ---: | ---: | ---: | --- |
| MiniDiT / Transformer | sample | `/hy-tmp/wan22_adaptive_threshold_mini_dit_cls_convpatch_3x12x8_d96_l2_bs128_speedup_20260706_171523` | 18 | 13 | 0.005919 | 18 |
| MiniDiT / Transformer | row | `/hy-tmp/wan22_adaptive_threshold_mini_dit_cls_convpatch_rowsplit_packed_d96_l2_bs128_speedup_20260706_171523` | 30 | 30 | 0.003765 | none |
| 5-feature gated MLP | sample | `/hy-tmp/wan22_adaptive_threshold_mlp_gated_5feature_range_speedup_samplesplit_long100_20260706_171523` | 72 | 52 | 0.008678 | 72 |
| 5-feature gated MLP | row | `/hy-tmp/wan22_adaptive_threshold_mlp_gated_5feature_range_speedup_rowsplit_long100_20260706_171523` | 100 | 93 | 0.007119 | none |

## Artifact Check

All four run roots contain:

- `best_model.pt`
- `best_model_checkpoint.pt`
- `final_model.pt`
- `final_model_checkpoint.pt`
- `val_predictions.csv`
- `epoch_metrics.csv`
- `epoch_metrics.jsonl`
- `config.json`
- `split.json`
- `model_summary.json`
- `metrics.json`

Symlinks were added under `experiment_results/` for the aggregate root and the four completed run roots.

## Notes

- The transformer/MiniDiT runs preserved the previous transformer architecture training settings and result-recording convention, with the only intended behavior change being the added speedup condition input and removal of the obsolete `target_oracle` training mode.
- The 5-feature runs used a 100-epoch budget as requested.
- Offline inverse-task MAE improved sharply after adding measured speedup as an input. This is expected because achieved speedup is strongly correlated with the fixed threshold label.
- Online adaptive SeaCache validation with these new checkpoints is still required before treating the checkpoints as deployment-ready.
