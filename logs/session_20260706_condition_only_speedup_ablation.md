# 2026-07-06 Condition-Only Speedup Ablations

## Scope

- Ran two requested 5-feature-network baseline checks with row split and the long100 training settings.
- The model input consumed no latent/features for either run.
- Added `--condition_inputs` to `adaptive_threshold_predictor.train_gate` and `ConditionOnlyAdaCacheGate` so condition-only baselines can ablate target PSNR while preserving the default three-condition behavior.
- No git commit was made.

## Runs

- Aggregate root: `/hy-tmp/wan22_condition_only_speedup_ablation_rowsplit_20260706_221900`
- Aggregate script: `/hy-tmp/wan22_condition_only_speedup_ablation_rowsplit_20260706_221900/commands/run_all.sh`
- Aggregate log: `/hy-tmp/wan22_condition_only_speedup_ablation_rowsplit_20260706_221900/logs/run_all.log`

| Run | Inputs | Root | Epochs | Best epoch | Best val MAE | Early stop | Params |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: |
| condition-only full | `timestep,target_psnr,target_speedup` | `/hy-tmp/wan22_adaptive_threshold_condition_only_fullcond_rowsplit_long100_20260706_221900` | 80 | 60 | 0.007347 | 80 | 12,929 |
| condition-only speedup-only | `timestep,target_speedup` | `/hy-tmp/wan22_adaptive_threshold_condition_only_speeduponly_rowsplit_long100_20260706_221900` | 100 | 85 | 0.007612 | none | 12,865 |

## Artifact Check

Both run roots contain:

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

Symlinks were added under `experiment_results/` for the aggregate root and both run roots.

## Interpretation

- The speedup-conditioned 5-feature row-split model had best val MAE `0.007119`.
- The full condition-only baseline reached `0.007347`, essentially the same offline inverse accuracy without latent/features.
- The speedup-only baseline reached `0.007612`, only slightly worse after removing target PSNR.
- This supports the hypothesis that most of the offline loss reduction comes from measured speedup itself; in this fixed-candidate SeaCache dataset, speedup nearly determines the threshold label.
