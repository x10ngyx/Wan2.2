# Session Log: Predictor Max-Condition Behavior Probe

Date: 2026-07-08

## Goal

Probe the adaptive threshold predictor behavior near max/high target speedup and
high target PSNR:

1. Fix a large `target_speedup`, vary latent/timestep/PSNR, and inspect
   predicted threshold.
2. Fix a large `target_psnr`, vary latent/timestep/speedup, and inspect
   predicted threshold.

Expected behavior: predicted threshold depends almost only on `target_speedup`.

## Environment

- Workdir: `/hy-tmp/work/Wan2.2`
- Conda env used: `/hy-tmp/miniconda3/envs/Wan2.2`
- GPU status:
  - Initial CPU-only probe: unavailable, `nvidia-smi` returned `No devices were found`.
  - Follow-up GPU probe: available, `NVIDIA A100 80GB PCIe`, `81920 MiB`, driver `570.211.01`; `torch.cuda.is_available()` returned `True`.

## Work Done

- Added reusable probe script:
  - `adaptive_threshold_predictor/probe_condition_sensitivity.py`
- Generated result bundle:
  - `/hy-tmp/wan22_predictor_max_condition_behavior_20260708`
- Added report:
  - `reports/report_predictor_max_condition_behavior.md`
  - `reports/report_max_speedup_psnr.md`
- Updated handoff index:
  - `PROGRESS.md`
- Ran GPU raw-latent MiniDiT intervention after user approved the corrected
  source-timestep-binding design.

## Results

Online MiniDiT trace analysis from:

```text
/hy-tmp/wan22_adaptive_seacache_mini_dit_rowsplit_speedup_sweep_50step_45f_480p_20260706_194715
```

- `3,600` predictor calls.
- Categorical mean R2 for threshold:
  - `target_speedup`: `0.999404`
  - `target_speedup + step_index`: `0.999557`
  - `target_speedup + sample_id`: `0.999601`
  - `target_psnr + target_speedup`: `0.999404`

Condition-only speedup-only intervention using:

```text
/hy-tmp/wan22_adaptive_threshold_condition_only_speeduponly_rowsplit_long100_20260706_221900/best_model_checkpoint.pt
```

- Fixed `target_speedup=3.5`, varying pseudo latent/timestep/PSNR:
  threshold range `0.000215`.
- Fixed `target_psnr=45`, varying pseudo latent/timestep/speedup:
  threshold range `0.689347`.

Conclusion: the current speedup-conditioned inverse-task predictors behave
effectively as `target_speedup -> threshold` mappers. Latent, timestep, and PSNR
contribute only small residual effects in the tested traces.

GPU raw-latent MiniDiT intervention:

```text
/hy-tmp/wan22_predictor_condition_sensitivity_probe_gpu_20260708
```

- Real train/val latents were sampled by source `step_index=0,12,24,36,49`.
- Each real latent was bound to its `source_timestep`.
- Random matched-normal latents were kept as an OOD appendix and swept over
  timesteps.
- Fixed `target_speedup=3.5` on source-bound real latents:
  - `n=100`, mean threshold `0.766872`, range `0.118354`.
  - Mean threshold by target PSNR dropped from `0.793165` at PSNR `18` to
    `0.723120` at PSNR `45`.
- Fixed `target_psnr=45` on source-bound real latents:
  - `n=160`, mean threshold `0.383865`, range `0.671022`.
  - Mean threshold increased from `0.100410` at speedup `1.1` to `0.723120`
    at speedup `3.5`.

Updated conclusion: speedup is the dominant control signal, but the actual
MiniDiT checkpoint is not literally independent of PSNR. At fixed high speedup,
higher requested PSNR lowers the predicted threshold.

## Incomplete / Follow-Up

- No Wan T2V video inference was run in this session; this was a predictor-only
  forward probe.
- A larger raw-latent probe can be run later by increasing
  `--real_latents_per_step` or expanding `--step_indices`.

## Validation

- Static compile passed:

```bash
/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m py_compile adaptive_threshold_predictor/probe_condition_sensitivity.py
```

- Result files written:
  - `/hy-tmp/wan22_predictor_max_condition_behavior_20260708/summary.json`
  - `/hy-tmp/wan22_predictor_max_condition_behavior_20260708/report.md`
  - `/hy-tmp/wan22_predictor_max_condition_behavior_20260708/online_trace_by_speedup.csv`
  - `/hy-tmp/wan22_predictor_max_condition_behavior_20260708/online_trace_by_psnr_speedup.csv`
  - `/hy-tmp/wan22_predictor_max_condition_behavior_20260708/online_trace_by_speedup_sample.csv`
  - `/hy-tmp/wan22_predictor_max_condition_behavior_20260708/online_trace_by_speedup_step.csv`
  - `/hy-tmp/wan22_predictor_max_condition_behavior_20260708/condition_only_intervention_rows.csv`
  - `/hy-tmp/wan22_predictor_condition_sensitivity_probe_gpu_20260708/summary.json`
  - `/hy-tmp/wan22_predictor_condition_sensitivity_probe_gpu_20260708/report.md`
  - `/hy-tmp/wan22_predictor_condition_sensitivity_probe_gpu_20260708/fixed_speedup_vary_latent_timestep_psnr.csv`
  - `/hy-tmp/wan22_predictor_condition_sensitivity_probe_gpu_20260708/fixed_psnr_vary_latent_timestep_speedup.csv`

## Final Report

- User requested a focused one-report summary named `report_max_speedup_psnr`.
- Created `reports/report_max_speedup_psnr.md` with the experiment settings,
  command, real source-bound result tables, random/OOD appendix, and final
  conclusion.
