# Predictor Max-Condition Behavior Probe

Date: 2026-07-08

## Question

We wanted to inspect the speedup-conditioned predictor near high target speedup
and high target PSNR settings:

1. Fix a large `target_speedup`, vary latent/timestep/PSNR, and observe the
   predicted threshold.
2. Fix a large `target_psnr`, vary latent/timestep/speedup, and observe the
   predicted threshold.

The expected behavior was that predicted threshold is controlled almost only by
`target_speedup`.

## Environment Notes

The initial probe session was not in GPU mode:

```text
nvidia-smi: No devices were found
torch.cuda.is_available(): False
```

Because arbitrary raw-latent MiniDiT probing is slow on CPU, the session used:

- real online MiniDiT predictor traces from the completed speedup sweep;
- a direct intervention on the speedup-only condition-only checkpoint;
- a reusable raw-latent MiniDiT probe script for later GPU-mode runs:
  `python -m adaptive_threshold_predictor.probe_condition_sensitivity`.

Artifacts are under:

```text
/hy-tmp/wan22_predictor_max_condition_behavior_20260708
```

A later session reran the raw-latent MiniDiT probe in GPU mode:

```text
GPU: NVIDIA A100 80GB PCIe, 81920 MiB, driver 570.211.01
torch.cuda.is_available(): True
```

GPU raw-latent probe artifacts are under:

```text
/hy-tmp/wan22_predictor_condition_sensitivity_probe_gpu_20260708
```

## Online MiniDiT Trace Check

Source:

```text
/hy-tmp/wan22_adaptive_seacache_mini_dit_rowsplit_speedup_sweep_50step_45f_480p_20260706_194715
```

This uses the speedup-conditioned MiniDiT row-split checkpoint and real Wan2.2
online inference latents. The 36 completed candidates contain `3,600`
predictor calls.

Categorical-mean R2 for predicted threshold:

| Grouping | R2 |
|---|---:|
| `target_speedup` | `0.999404` |
| `target_speedup + step_index` | `0.999557` |
| `target_speedup + sample_id` | `0.999601` |
| `target_psnr + target_speedup` | `0.999404` |

Mean threshold by target speedup:

| Target speedup | Calls | Mean threshold | Std | Min | Max | Range |
|---:|---:|---:|---:|---:|---:|---:|
| `1.4` | `600` | `0.146937` | `0.000949` | `0.144531` | `0.150391` | `0.005859` |
| `1.7` | `600` | `0.216816` | `0.002672` | `0.211914` | `0.234375` | `0.022461` |
| `2.0` | `600` | `0.308385` | `0.001376` | `0.296875` | `0.310547` | `0.013672` |
| `2.2` | `600` | `0.359909` | `0.005188` | `0.339844` | `0.369141` | `0.029297` |
| `2.5` | `600` | `0.430332` | `0.002235` | `0.423828` | `0.439453` | `0.015625` |
| `2.8` | `600` | `0.499831` | `0.003139` | `0.498047` | `0.511719` | `0.013672` |

The online trace does not contain the exact same high `target_speedup` crossed
with multiple `target_psnr` values, but it does contain different prompts,
latents, steps, branches, stages, and PSNR targets across the speedup sweep.
Adding `target_psnr` to `target_speedup` does not improve R2 beyond
`target_speedup` alone in this trace table.

## Speedup-Only Condition-Only Intervention

Checkpoint:

```text
/hy-tmp/wan22_adaptive_threshold_condition_only_speeduponly_rowsplit_long100_20260706_221900/best_model_checkpoint.pt
```

This checkpoint consumes only `(timestep, target_speedup)`. It is therefore a
control for the hypothesis that the inverse task has reduced to a
speedup-to-threshold mapping. It cannot consume latent or target PSNR, so the
latent labels in this probe are explicit invariance proxies.

### Fixed Large Speedup

Fixed `target_speedup=3.5`; varied pseudo-latent label, timestep
`0.0/0.25/0.5/0.75/1.0`, and target PSNR `18/22/28/35/45`.

Overall:

| Rows | Mean threshold | Std | Min | Max | Range |
|---:|---:|---:|---:|---:|---:|
| `75` | `0.790103` | `0.000079` | `0.789999` | `0.790214` | `0.000215` |

By target PSNR:

| Target PSNR | Rows | Mean threshold | Range |
|---:|---:|---:|---:|
| `18` | `15` | `0.790103` | `0.000215` |
| `22` | `15` | `0.790103` | `0.000215` |
| `28` | `15` | `0.790103` | `0.000215` |
| `35` | `15` | `0.790103` | `0.000215` |
| `45` | `15` | `0.790103` | `0.000215` |

Interpretation: changing PSNR and pseudo latent did not move the threshold.
Only tiny timestep-level residual variation remains.

### Fixed Large PSNR

Fixed `target_psnr=45`; varied pseudo-latent label, timestep, and target
speedup `1.1/1.4/1.7/2.0/2.5/2.8/3.2/3.5`.

Overall:

| Rows | Mean threshold | Std | Min | Max | Range |
|---:|---:|---:|---:|---:|---:|
| `120` | `0.389386` | `0.227387` | `0.100867` | `0.790214` | `0.689347` |

By target speedup:

| Target speedup | Rows | Mean threshold | Range |
|---:|---:|---:|---:|
| `1.1` | `15` | `0.101020` | `0.000388` |
| `1.4` | `15` | `0.147951` | `0.000620` |
| `1.7` | `15` | `0.224550` | `0.000697` |
| `2.0` | `15` | `0.295912` | `0.000163` |
| `2.5` | `15` | `0.421667` | `0.000657` |
| `2.8` | `15` | `0.503028` | `0.000449` |
| `3.2` | `15` | `0.630856` | `0.000751` |
| `3.5` | `15` | `0.790103` | `0.000215` |

Interpretation: fixing a high PSNR and changing speedup moves threshold across
almost the entire allowed range `[0.10, 0.80]`.

## GPU Raw-Latent MiniDiT Probe

Checkpoint:

```text
/hy-tmp/wan22_adaptive_threshold_mini_dit_cls_convpatch_rowsplit_packed_d96_l2_bs128_speedup_20260706_171523/best_model_checkpoint.pt
```

Packed latent cache:

```text
/hy-tmp/wan22_adaptive_threshold_raw_latent_packed_cache_candidate_inverse_fp16_20260629_221805
```

The probe binds each real train/validation latent to its source timestep. This
avoids pairing a step-0 latent with an unrelated timestep condition. Real
latents were sampled by source step:

```text
step_index = 0, 12, 24, 36, 49
real_latents_per_step = 2 for train and 2 for validation
```

This gives `20` real latents. Three matched-normal random latents were also
included as an OOD appendix and swept over normalized timestep
`0.0/0.25/0.5/0.75/1.0`; random rows are not used for the main readout.

### Fixed Large Speedup, Source-Bound Real Latents

Fixed:

```text
target_speedup = 3.5
```

Varied target PSNR:

```text
18, 22, 28, 35, 45
```

Real-bound result:

| Rows | Mean threshold | Std | Min | Max | Range |
|---:|---:|---:|---:|---:|---:|
| `100` | `0.766872` | `0.035692` | `0.680666` | `0.799020` | `0.118354` |

By target PSNR:

| Target PSNR | Rows | Mean threshold | Std | Min | Max | Range |
|---:|---:|---:|---:|---:|---:|---:|
| `18` | `20` | `0.793165` | `0.011498` | `0.758094` | `0.799020` | `0.040926` |
| `22` | `20` | `0.787356` | `0.021701` | `0.727528` | `0.798909` | `0.071381` |
| `28` | `20` | `0.776094` | `0.031604` | `0.703981` | `0.798477` | `0.094497` |
| `35` | `20` | `0.754624` | `0.030750` | `0.685447` | `0.795115` | `0.109668` |
| `45` | `20` | `0.723120` | `0.025657` | `0.680666` | `0.771310` | `0.090644` |

By source step:

| Source step | Rows | Mean threshold | Std | Min | Max | Range |
|---:|---:|---:|---:|---:|---:|---:|
| `0` | `20` | `0.771999` | `0.034689` | `0.709604` | `0.798574` | `0.088970` |
| `12` | `20` | `0.771725` | `0.034754` | `0.700167` | `0.798912` | `0.098745` |
| `24` | `20` | `0.772794` | `0.034983` | `0.695052` | `0.799020` | `0.103968` |
| `36` | `20` | `0.740021` | `0.041465` | `0.680666` | `0.798909` | `0.118243` |
| `49` | `20` | `0.777820` | `0.017576` | `0.748764` | `0.798605` | `0.049842` |

Readout: at high `target_speedup=3.5`, MiniDiT predicts high thresholds overall,
but target PSNR still has a visible secondary effect. The mean threshold drops
from `0.793165` at PSNR `18` to `0.723120` at PSNR `45`.

### Fixed Large PSNR, Source-Bound Real Latents

Fixed:

```text
target_psnr = 45
```

Varied target speedup:

```text
1.1, 1.4, 1.7, 2.0, 2.5, 2.8, 3.2, 3.5
```

Real-bound result:

| Rows | Mean threshold | Std | Min | Max | Range |
|---:|---:|---:|---:|---:|---:|
| `160` | `0.383865` | `0.214063` | `0.100288` | `0.771310` | `0.671022` |

By target speedup:

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

By source step:

| Source step | Rows | Mean threshold | Std | Min | Max | Range |
|---:|---:|---:|---:|---:|---:|---:|
| `0` | `32` | `0.382741` | `0.214730` | `0.100295` | `0.709604` | `0.609309` |
| `12` | `32` | `0.382965` | `0.214823` | `0.100288` | `0.727259` | `0.626971` |
| `24` | `32` | `0.383945` | `0.216670` | `0.100295` | `0.761217` | `0.660923` |
| `36` | `32` | `0.382859` | `0.215299` | `0.100321` | `0.771310` | `0.670989` |
| `49` | `32` | `0.386816` | `0.222390` | `0.100601` | `0.760653` | `0.660052` |

Readout: with high `target_psnr=45`, target speedup dominates the prediction.
The mean threshold rises monotonically from `0.100410` at speedup `1.1` to
`0.723120` at speedup `3.5`. Source-step means are nearly identical because
each step group contains the full speedup sweep.

### Random/OOD Appendix

Random matched-normal latents were swept over the requested timesteps:

| Probe | Rows | Mean threshold | Std | Min | Max | Range |
|---|---:|---:|---:|---:|---:|---:|
| Fixed speedup `3.5` | `75` | `0.738182` | `0.024218` | `0.707627` | `0.790733` | `0.083106` |
| Fixed PSNR `45` | `120` | `0.378582` | `0.211140` | `0.103033` | `0.738240` | `0.635208` |

These rows are useful as an OOD sanity check but are not mixed into the main
source-bound conclusion.

## Conclusion

The observed behavior mostly matches the expectation, with one important
qualification from the GPU raw-latent probe:

- For the real online MiniDiT predictor, `target_speedup` alone explains
  `99.94%` of predicted-threshold variance.
- For the speedup-only condition-only control, fixed high speedup produces an
  essentially constant threshold even while PSNR and pseudo latent vary.
- For the actual MiniDiT raw-latent checkpoint, fixed high PSNR plus varying
  speedup changes threshold from about `0.10` to about `0.72` on source-bound
  real latents.
- At fixed high `target_speedup=3.5`, MiniDiT does still respond to target PSNR:
  mean threshold decreases from about `0.793` at PSNR `18` to about `0.723` at
  PSNR `45`.

The practical readout is that the current speedup-conditioned inverse-task
predictor is strongly dominated by `target_speedup`, especially when speedup is
varied directly. It is not literally independent of PSNR: under a fixed very
large speedup target, higher requested PSNR pushes the MiniDiT threshold lower.
Latent/source-step effects are smaller than the speedup sweep and are partly
entangled with PSNR in the fixed-speedup table.
