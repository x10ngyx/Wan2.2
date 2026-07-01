# Session Log: Gated MLP Range-Constrained Output Retrain

Date: 2026-06-30

## Request

Fix the 5-feature gated MLP output mismatch. The previous MLP head emitted a
direct sigmoid value in `[0, 1]`; it should match the Transformer/MiniDiT-style
mapping:

```text
threshold = min_threshold + sigmoid(raw) * (max_threshold - min_threshold)
```

with default range `[0.10, 0.80]`. Retrain:

- sample split, 30 epochs
- row split, 30 epochs
- row split, 100 epochs

and compare against the previous direct-sigmoid 5-feature results.

## Code Changes

- `adaptive_threshold_predictor/models.py`
  - MLP-family threshold heads now emit raw logits and apply scaled sigmoid
    mapping to `[min_threshold, max_threshold]`.
  - Updated:
    - `ImprovedAdaCacheGate`
    - `CachedFeatureAdaCacheGate`
    - `GatedFeatureFusionAdaCacheGate`
    - `CachedGatedFeatureAdaCacheGate`
    - `GatedMultiFeatureAdaCacheGate`
    - `ConditionOnlyAdaCacheGate`
- `adaptive_threshold_predictor/train_gate.py`
  - `build_model()` now passes `--min_threshold` and `--max_threshold` into MLP
    and condition-only model constructors.
- `adaptive_threshold_predictor/README.md`
  - Documented the scaled-sigmoid threshold mapping and default `[0.10, 0.80]`
    range.
- `reports/report_gated_multifeature_mlp_architecture.md`
  - Replaced stale direct-`Sigmoid` `[0,1]` architecture wording.
  - Added range-constrained retrain results and comparison table.
- `PROGRESS.md`
  - Added the range-constrained retrain handoff section.

## Validation

- `py_compile` passed for:
  - `adaptive_threshold_predictor/models.py`
  - `adaptive_threshold_predictor/train_gate.py`
  - `adaptive_threshold_predictor/data.py`
- Random forward smoke for five-feature `CachedGatedFeatureAdaCacheGate` produced
  `[B, 1]` predictions inside `[0.10, 0.80]`.
- Temporary CPU smoke output was removed:
  - `/hy-tmp/wan22_adaptive_threshold_mlp_gated_5feature_range_smoke_20260630_0345`

## Training Outputs

All three GPU trainings completed.

| Run | Root |
|---|---|
| sample split 30 | `/hy-tmp/wan22_adaptive_threshold_mlp_gated_5feature_range_samplesplit_20260630_035000` |
| row split 30 | `/hy-tmp/wan22_adaptive_threshold_mlp_gated_5feature_range_rowsplit_gpu_20260630_035000` |
| row split 100 | `/hy-tmp/wan22_adaptive_threshold_mlp_gated_5feature_range_rowsplit_gpu_long100_20260630_035000` |

Matching symlinks were created under `experiment_results/`.

Each result root contains:

- `commands/launch_train.sh`
- `logs/train.log`
- `metrics.json`
- `epoch_metrics.jsonl`
- `epoch_metrics.csv`
- `val_predictions.csv`
- best/final model and checkpoint files

## Results

| Run | Split | Epochs Run | Best Epoch | Best Val MAE | Best Val Loss | Final Val MAE | Best Prediction Range |
|---|---|---:|---:|---:|---:|---:|---|
| range sample split 30 | sample | 14 / 30, early stopped | 9 | `0.1143567288` | `0.1049280047` | `0.1278297383` | `[0.1023, 0.7303]` |
| range row split 30 | row | 30 / 30 | 30 | `0.0770497653` | `0.0678418150` | `0.0770497653` | `[0.1040, 0.7743]` |
| range row split 100 | row | 100 / 100 | 98 | `0.0610311001` | `0.0522424302` | `0.0610505000` | `[0.1010, 0.7985]` |

Mean validation gate weights:

| Run | latent_pool | temporal_mean | temporal_var | frame_diff_mean | frame_diff_var |
|---|---:|---:|---:|---:|---:|
| sample split 30 | `0.5464` | `0.1839` | `0.0945` | `0.0986` | `0.0766` |
| row split 30 | `0.4781` | `0.1967` | `0.1267` | `0.0998` | `0.0987` |
| row split 100 | `0.3987` | `0.2533` | `0.0942` | `0.0962` | `0.1576` |

Comparison against previous direct-sigmoid `[0,1]` 5-feature runs:

| Split / Budget | Previous Best Val MAE | Range-Constrained Best Val MAE | Delta |
|---|---:|---:|---:|
| sample split 30 | `0.1142528785` | `0.1143567288` | `+0.0001038503` |
| row split 30 | `0.0756697811` | `0.0770497653` | `+0.0013799842` |
| row split 100 | `0.0601118673` | `0.0610311001` | `+0.0009192327` |

## Interpretation

- The output mismatch is fixed and predictions are constrained to `[0.10, 0.80]`.
- Offline threshold MAE did not improve. Sample split is effectively unchanged;
  row split is slightly worse.
- Row split still benefits from longer training, improving from `0.07705` at 30
  epochs to `0.06103` at 100 epochs.
- The range-constrained gated MLP remains behind the MiniDiT row-split reference
  MAE `0.0380019387`.

## Cleanup / Status

- No commit was made.
- Superseded temporary smoke output was removed.
- Final validation was run after documentation updates.
