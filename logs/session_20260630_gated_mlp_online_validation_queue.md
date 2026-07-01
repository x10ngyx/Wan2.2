# Session Log: 5-Feature Gated MLP Online Validation Queue

Date: 2026-06-30

## Request

The Transformer/MiniDiT adaptive predictor is currently being evaluated online
inside `adaptive_seacache_wan22`. Start an identical validation experiment for
the 5-feature gated MLP architecture.

## Existing Transformer Online Experiment

- tmux session: `wan22_adaptive_mini_dit_split_20260630_025328`
- result root:
  `/hy-tmp/wan22_adaptive_seacache_mini_dit_split_compare_50step_45f_480p_20260630_025328`
- runner:
  `experiments/adaptive_seacache_mini_dit_split_compare_50step_45f_480p/run_batch.py`
- protocol:
  - 2 splits: `sample_split`, `row_split`
  - 2 target PSNRs: `22`, `28`
  - 2 datasets: `VBench10`, `OpenVid100 train`
  - 3 prompts per dataset
  - total: 24 candidates
- The runner reuses existing baselines and loads the WanT2V pipeline once.

At handoff time, the Transformer run had 22 completed summary rows and was
running candidate 23/24.

## Implementation

The online adaptive SeaCache code previously supported:

- MiniDiT raw-latent predictor
- legacy single-feature `CachedFeatureAdaCacheGate`

The 5-feature checkpoint is `CachedGatedFeatureAdaCacheGate`, so online support
was added.

Changed files:

- `adaptive_seacache_wan22/cache.py`
  - Added `mlp_gated` auto-detection for checkpoints with `fusion.*` state keys.
  - Added metadata loading for `feature_sets`, `hidden_dim`,
    `feature_embedding_dim`, `psnr_min/max`, and `min/max_threshold`.
  - Added online extraction for:
    - `latent_pool`
    - `temporal_mean`
    - `temporal_var`
    - `frame_diff_mean`
    - `frame_diff_var`
  - Preserved old single-feature MLP and MiniDiT paths.
- `adaptive_seacache_wan22/generate_t2v.py`
  - Added `mlp_gated` as an explicit `--adaptive_model_type` choice.
- Added launch helper:
  - `experiments/adaptive_seacache_mlp_gated_5feature_split_compare_50step_45f_480p/run_tmux.sh`

## Validation

- `py_compile` passed for:
  - `adaptive_seacache_wan22/cache.py`
  - `adaptive_seacache_wan22/generate_t2v.py`
  - `adaptive_threshold_predictor/models.py`
- Loaded the 5-feature sample-split checkpoint with `model_type=auto`; it
  resolved to `mlp_gated`.
- Confirmed online metadata:
  - feature sets:
    `latent_pool temporal_mean temporal_var frame_diff_mean frame_diff_var`
  - hidden dim: `64`
  - feature embedding dim: `64`
  - threshold range: `[0.1, 0.8]`
- Random latent forward prediction succeeded and returned a threshold inside
  `[0.1, 0.8]`.
- CPU validation of the runner with 5-feature checkpoints produced the expected
  24 candidates and reusable baselines.

## Queued Experiment

The experiment was queued rather than launched concurrently, because the
Transformer validation is already using the single A100.

- tmux session: `wan22_adaptive_mlp_gated5_split_20260630_050727`
- waits for tmux session:
  `wan22_adaptive_mini_dit_split_20260630_025328`
- result root:
  `/hy-tmp/wan22_adaptive_seacache_mlp_gated_5feature_range_split_compare_50step_45f_480p_20260630_050727`
- symlink:
  `experiment_results/wan22_adaptive_seacache_mlp_gated_5feature_range_split_compare_50step_45f_480p_20260630_050727`

5-feature checkpoints:

- sample split:
  `/hy-tmp/wan22_adaptive_threshold_mlp_gated_5feature_range_samplesplit_20260630_035000/best_model_checkpoint.pt`
- row split:
  `/hy-tmp/wan22_adaptive_threshold_mlp_gated_5feature_range_rowsplit_gpu_long100_20260630_035000/best_model_checkpoint.pt`

The queued command passes:

- `--target_psnrs '22 28'`
- `--prompt_count 3`
- `--sample_split_json /hy-tmp/wan22_adaptive_threshold_mlp_gated_5feature_range_samplesplit_20260630_035000/split.json`
- `--adaptive_min_threshold 0.10`
- `--adaptive_max_threshold 0.80`
- `--resume_existing`

## Monitoring

Transformer run:

```bash
tmux attach -t wan22_adaptive_mini_dit_split_20260630_025328
tail -f /hy-tmp/wan22_adaptive_seacache_mini_dit_split_compare_50step_45f_480p_20260630_025328/logs/runner.log
```

Queued 5-feature run:

```bash
tmux attach -t wan22_adaptive_mlp_gated5_split_20260630_050727
tail -f /hy-tmp/wan22_adaptive_seacache_mlp_gated_5feature_range_split_compare_50step_45f_480p_20260630_050727/logs/runner.log
```

## Status

- 5-feature validation is queued and waiting for the Transformer tmux session to
  exit.
- No commit was made.

## Completion Update

The queued 5-feature gated MLP validation completed successfully.

- tmux session exited.
- `summary.csv` has 24 completed candidate rows.
- `failed/` is empty.
- Runner log ended with:
  `Completed experiment: /hy-tmp/wan22_adaptive_seacache_mlp_gated_5feature_range_split_compare_50step_45f_480p_20260630_050727`

Result files:

- `/hy-tmp/wan22_adaptive_seacache_mlp_gated_5feature_range_split_compare_50step_45f_480p_20260630_050727/results/summary.csv`
- `/hy-tmp/wan22_adaptive_seacache_mlp_gated_5feature_range_split_compare_50step_45f_480p_20260630_050727/results/aggregate_by_dataset_model_target.csv`

Aggregate results:

| Dataset | Split | Target | Overall Speedup | Mean PSNR | Target Error | Mean Threshold |
|---|---|---:|---:|---:|---:|---:|
| OpenVid train | row | 22 | `2.523x` | `21.970` | `-0.030` | `0.488` |
| OpenVid train | row | 28 | `1.773x` | `26.095` | `-1.905` | `0.279` |
| OpenVid train | sample | 22 | `2.559x` | `22.858` | `+0.858` | `0.495` |
| OpenVid train | sample | 28 | `1.718x` | `27.365` | `-0.635` | `0.291` |
| VBench10 | row | 22 | `2.065x` | `20.810` | `-1.190` | `0.353` |
| VBench10 | row | 28 | `1.627x` | `24.484` | `-3.516` | `0.201` |
| VBench10 | sample | 22 | `2.229x` | `17.159` | `-4.841` | `0.358` |
| VBench10 | sample | 28 | `1.539x` | `25.460` | `-2.540` | `0.184` |

Comparison against the same-protocol MiniDiT run:

| Dataset | Split | Target | 5-feature Speedup | MiniDiT Speedup | 5-feature PSNR | MiniDiT PSNR |
|---|---|---:|---:|---:|---:|---:|
| OpenVid train | row | 22 | `2.523x` | `2.447x` | `21.970` | `23.007` |
| OpenVid train | row | 28 | `1.773x` | `1.633x` | `26.095` | `27.710` |
| OpenVid train | sample | 22 | `2.559x` | `2.598x` | `22.858` | `22.151` |
| OpenVid train | sample | 28 | `1.718x` | `1.794x` | `27.365` | `29.019` |
| VBench10 | row | 22 | `2.065x` | `2.068x` | `20.810` | `20.466` |
| VBench10 | row | 28 | `1.627x` | `1.539x` | `24.484` | `25.469` |
| VBench10 | sample | 22 | `2.229x` | `2.113x` | `17.159` | `16.737` |
| VBench10 | sample | 28 | `1.539x` | `1.582x` | `25.460` | `23.796` |

Immediate read:

- 5-feature MLP is not uniformly worse online despite weaker offline MAE.
- OpenVid target 28 favors MiniDiT on PSNR.
- VBench10 sample-split target 28 favors 5-feature MLP on PSNR in this small
  3-prompt pilot.
- Both predictors still show weak target control on VBench10, especially target
  22 under sample split.

After completion, `nvidia-smi` returned `No devices were found`; the GPU appears
to have been disabled or detached after the run.
