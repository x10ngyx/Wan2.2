# Max Speedup / PSNR Predictor Probe

Date: 2026-07-08

## Purpose

This probe checks how the speedup-conditioned MiniDiT threshold predictor behaves
when either `target_speedup` or `target_psnr` is held at a large value.

The original hypothesis was:

```text
predicted threshold is controlled almost only by target_speedup
```

The final readout is more precise:

```text
target_speedup is the dominant control signal, but the MiniDiT checkpoint is not
strictly speedup-only. At fixed high target_speedup, increasing target_psnr still
lowers the predicted threshold.
```

## Environment

GPU was available for this run:

```text
GPU: NVIDIA A100 80GB PCIe
GPU memory: 81920 MiB
Driver: 570.211.01
torch.cuda.is_available(): True
```

The probe was predictor-only. It did not run Wan2.2 video inference, cache
runtime, FFmpeg, or PSNR evaluation.

## Model And Data

Predictor checkpoint:

```text
/hy-tmp/wan22_adaptive_threshold_mini_dit_cls_convpatch_rowsplit_packed_d96_l2_bs128_speedup_20260706_171523/best_model_checkpoint.pt
```

Packed raw latent cache:

```text
/hy-tmp/wan22_adaptive_threshold_raw_latent_packed_cache_candidate_inverse_fp16_20260629_221805
```

Output root:

```text
/hy-tmp/wan22_predictor_condition_sensitivity_probe_gpu_20260708
```

Main output files:

```text
/hy-tmp/wan22_predictor_condition_sensitivity_probe_gpu_20260708/fixed_speedup_vary_latent_timestep_psnr.csv
/hy-tmp/wan22_predictor_condition_sensitivity_probe_gpu_20260708/fixed_psnr_vary_latent_timestep_speedup.csv
/hy-tmp/wan22_predictor_condition_sensitivity_probe_gpu_20260708/summary.json
/hy-tmp/wan22_predictor_condition_sensitivity_probe_gpu_20260708/report.md
```

## Sampling Design

The important correction in this run is that real train/validation latents are
bound to their original source timestep. We do not pair a step-0 latent with an
unrelated timestep such as `0.75`.

Requested source steps:

```text
step_index = 0, 12, 24, 36, 49
```

Real latent sampling:

| Split | Latents per step | Steps | Total real latents |
|---|---:|---:|---:|
| train | `2` | `5` | `10` |
| validation | `2` | `5` | `10` |
| total | - | - | `20` |

Each real latent uses its own:

```text
source_step_index
source_timestep
```

Random/OOD appendix:

| Latent type | Count | Timestep handling | Used in main conclusion |
|---|---:|---|---|
| matched-normal random latent | `3` | swept over `0.0, 0.25, 0.5, 0.75, 1.0` | no |

Random latents are included only as an OOD sanity check and are not mixed into
the main source-bound real-latent tables.

## Command

```bash
/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m adaptive_threshold_predictor.probe_condition_sensitivity \
  --out_dir /hy-tmp/wan22_predictor_condition_sensitivity_probe_gpu_20260708 \
  --device cuda \
  --bind_real_latent_timestep \
  --step_indices 0,12,24,36,49 \
  --real_latents_per_step 2 \
  --num_random_latents 3 \
  --condition_batch_size 4 \
  --fixed_speedup 3.5 \
  --fixed_psnr 45.0 \
  --psnrs 18,22,28,35,45 \
  --speedups 1.1,1.4,1.7,2.0,2.5,2.8,3.2,3.5
```

## Experiment 1: Fixed High Target Speedup

Fixed:

```text
target_speedup = 3.5
```

Varied:

```text
target_psnr = 18, 22, 28, 35, 45
```

Real source-bound latent count:

```text
20 real latents * 5 target_psnr values = 100 predictions
```

### Overall

| Rows | Mean threshold | Std | Min | Max | Range |
|---:|---:|---:|---:|---:|---:|
| `100` | `0.766872` | `0.035692` | `0.680666` | `0.799020` | `0.118354` |

### By Target PSNR

| Target PSNR | Rows | Mean threshold | Std | Min | Max | Range |
|---:|---:|---:|---:|---:|---:|---:|
| `18` | `20` | `0.793165` | `0.011498` | `0.758094` | `0.799020` | `0.040926` |
| `22` | `20` | `0.787356` | `0.021701` | `0.727528` | `0.798909` | `0.071381` |
| `28` | `20` | `0.776094` | `0.031604` | `0.703981` | `0.798477` | `0.094497` |
| `35` | `20` | `0.754624` | `0.030750` | `0.685447` | `0.795115` | `0.109668` |
| `45` | `20` | `0.723120` | `0.025657` | `0.680666` | `0.771310` | `0.090644` |

### By Source Step

| Source step index | Rows | Mean threshold | Std | Min | Max | Range |
|---:|---:|---:|---:|---:|---:|---:|
| `0` | `20` | `0.771999` | `0.034689` | `0.709604` | `0.798574` | `0.088970` |
| `12` | `20` | `0.771725` | `0.034754` | `0.700167` | `0.798912` | `0.098745` |
| `24` | `20` | `0.772794` | `0.034983` | `0.695052` | `0.799020` | `0.103968` |
| `36` | `20` | `0.740021` | `0.041465` | `0.680666` | `0.798909` | `0.118243` |
| `49` | `20` | `0.777820` | `0.017576` | `0.748764` | `0.798605` | `0.049842` |

### Readout

At fixed high `target_speedup=3.5`, predicted thresholds are generally high.
However, the predictor is not independent of PSNR. Increasing the requested
PSNR makes the predictor more conservative:

```text
target_psnr 18 -> mean threshold 0.793165
target_psnr 45 -> mean threshold 0.723120
delta = -0.070045
```

This is a visible secondary PSNR effect under a fixed high speed target.

## Experiment 2: Fixed High Target PSNR

Fixed:

```text
target_psnr = 45
```

Varied:

```text
target_speedup = 1.1, 1.4, 1.7, 2.0, 2.5, 2.8, 3.2, 3.5
```

Real source-bound latent count:

```text
20 real latents * 8 target_speedup values = 160 predictions
```

### Overall

| Rows | Mean threshold | Std | Min | Max | Range |
|---:|---:|---:|---:|---:|---:|
| `160` | `0.383865` | `0.214063` | `0.100288` | `0.771310` | `0.671022` |

### By Target Speedup

| Target speedup | Rows | Mean threshold | Std | Min | Max | Range |
|---:|---:|---:|---:|---:|---:|---:|
| `1.1` | `20` | `0.100410` | `0.000244` | `0.100288` | `0.101138` | `0.000850` |
| `1.4` | `20` | `0.146739` | `0.001571` | `0.140779` | `0.148760` | `0.007980` |
| `1.7` | `20` | `0.220661` | `0.004222` | `0.216354` | `0.233073` | `0.016718` |
| `2.0` | `20` | `0.310053` | `0.004283` | `0.295298` | `0.316918` | `0.021620` |
| `2.5` | `20` | `0.431896` | `0.004671` | `0.415964` | `0.438883` | `0.022919` |
| `2.8` | `20` | `0.502148` | `0.002471` | `0.497883` | `0.508482` | `0.010598` |
| `3.2` | `20` | `0.635896` | `0.005809` | `0.622533` | `0.647415` | `0.024882` |
| `3.5` | `20` | `0.723120` | `0.025657` | `0.680666` | `0.771310` | `0.090644` |

### By Source Step

| Source step index | Rows | Mean threshold | Std | Min | Max | Range |
|---:|---:|---:|---:|---:|---:|---:|
| `0` | `32` | `0.382741` | `0.214730` | `0.100295` | `0.709604` | `0.609309` |
| `12` | `32` | `0.382965` | `0.214823` | `0.100288` | `0.727259` | `0.626971` |
| `24` | `32` | `0.383945` | `0.216670` | `0.100295` | `0.761217` | `0.660923` |
| `36` | `32` | `0.382859` | `0.215299` | `0.100321` | `0.771310` | `0.670989` |
| `49` | `32` | `0.386816` | `0.222390` | `0.100601` | `0.760653` | `0.660052` |

### Readout

At fixed high `target_psnr=45`, `target_speedup` is the dominant control signal.
The mean threshold rises monotonically with requested speedup:

```text
target_speedup 1.1 -> mean threshold 0.100410
target_speedup 3.5 -> mean threshold 0.723120
delta = +0.622710
```

The source-step means are very close because each source-step group contains the
same full speedup sweep.

## Random/OOD Appendix

The random matched-normal latents are not mixed into the main real-latent
conclusion.

| Probe | Rows | Mean threshold | Std | Min | Max | Range |
|---|---:|---:|---:|---:|---:|---:|
| Fixed `target_speedup=3.5` | `75` | `0.738182` | `0.024218` | `0.707627` | `0.790733` | `0.083106` |
| Fixed `target_psnr=45` | `120` | `0.378582` | `0.211140` | `0.103033` | `0.738240` | `0.635208` |

## Conclusion

The hypothesis that threshold is controlled mostly by `target_speedup` is
supported, but the stronger statement that threshold is only controlled by
`target_speedup` is not supported for the MiniDiT checkpoint.

Key conclusions:

1. `target_speedup` is the dominant control signal.
   - At fixed `target_psnr=45`, increasing speedup from `1.1` to `3.5` changes
     mean threshold from `0.100410` to `0.723120`.
   - The total real-latent threshold range is `0.671022`.

2. `target_psnr` has a visible secondary effect at fixed high speedup.
   - At fixed `target_speedup=3.5`, increasing PSNR from `18` to `45` lowers
     mean threshold from `0.793165` to `0.723120`.
   - The fixed-speedup real-latent threshold range is `0.118354`.

3. Source timestep and latent identity are smaller effects than the speedup
   sweep.
   - Real latents were bound to their source timestep to avoid OOD
     timestep-latent pairs.
   - The source-step grouping does not show a control effect comparable to the
     speedup sweep.

Short version:

```text
The predictor is speedup-dominated, not speedup-only.
```
