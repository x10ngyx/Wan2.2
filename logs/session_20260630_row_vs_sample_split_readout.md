# Session Log: Row Split vs Sample Split Readout

Date: 2026-06-30

## What I Did

- Compared the row-split and sample-split training and online inference
  outcomes for MiniDiT and 5-feature gated MLP predictors.

## Main Findings

- Row split is an interpolation diagnostic, not a held-out-sample
  generalization metric.
- MiniDiT benefits much more from row split than the 5-feature MLP, indicating
  stronger use of raw latent structure when source-video identity is shared.
- Online results do not consistently improve under row split, so row-split MAE
  is insufficient for choosing a deployed adaptive threshold predictor.
- Sample split is the more relevant offline metric, but the current
  `candidate_inverse` training objective still does not align well enough with
  online target-PSNR control.

## Changes

- Appended a short note to `PROGRESS.md`.
- Added this session log.

## Validation

- No code changes or experiments were run.
- No commit was made.
