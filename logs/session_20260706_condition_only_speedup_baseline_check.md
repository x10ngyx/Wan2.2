# Session Log: Condition-Only Speedup Baseline Check

Date: 2026-07-06

## Scope

Checked whether the recent predictor ablation followed the two requested condition-only baseline plans:

- `(timestep, target_psnr, target_speedup) -> threshold`, no latent/features.
- `(timestep, target_speedup) -> threshold`, no target PSNR and no latent/features.

## Findings

- Aggregate root checked: `/hy-tmp/wan22_condition_only_speedup_ablation_rowsplit_20260706_221900`.
- Full-condition run: `/hy-tmp/wan22_adaptive_threshold_condition_only_fullcond_rowsplit_long100_20260706_221900`.
- Speedup-only run: `/hy-tmp/wan22_adaptive_threshold_condition_only_speeduponly_rowsplit_long100_20260706_221900`.
- Both launch scripts use `--control_mode condition_only`.
- Full-condition launch uses `--condition_inputs timestep target_psnr target_speedup`; model summary confirms `ConditionOnlyAdaCacheGate` with `Linear(in_features=3, ...)`.
- Speedup-only launch uses `--condition_inputs timestep target_speedup`; model summary confirms `ConditionOnlyAdaCacheGate` with `Linear(in_features=2, ...)`.
- Code path in `adaptive_threshold_predictor/train_gate.py` instantiates `ConditionOnlyAdaCacheGate` and forwards only `timestep`, `target_psnr`, and `target_speedup` in condition-only mode; latent/features from the cached dataset are not consumed by the model.
- Full-condition result completed normally: best val MAE `0.007347` at epoch `60`, early stopped at epoch `80`, and wrote final checkpoints plus `val_predictions.csv`.
- Speedup-only result reached best val MAE `0.007836` at epoch `63`, but did not fully finalize: no `saved:` line, no `final_model*`, and no `val_predictions.csv`.

## Files Updated

- `PROGRESS.md`
- `logs/session_20260706_condition_only_speedup_baseline_check.md`

## Validation

Used launch scripts, configs, model summaries, epoch metric CSVs, logs, and the relevant `train_gate.py` / `models.py` code paths. No training or GPU inference was run.

## Remaining Work

Rerun or resume the speedup-only condition baseline to normal completion if a fully compliant archive is required.
