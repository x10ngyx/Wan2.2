# 2026-06-16 Adaptive Predictor Report Architecture Section

## Summary

- Continued `reports/report_adaptive_predictor.md`.
- Added section `2. 网络架构`.
- Also provided a short PPT-ready summary in the assistant response.

## Content Added

- Current task definition: timestep / SeaCache single-threshold prediction.
- Input/output table for raw latent and cached-feature paths.
- PSNR normalization.
- Two-branch network structure:
  - feature branch;
  - condition branch;
  - prediction head.
- Parameter counts for the feature model and condition-only control.
- Five latent-derived feature definitions.
- Cached-feature model module shapes.
- Control modes.
- Current default architecture configuration.

## Follow-up Revision

- Expanded the architecture section after user feedback.
- Added explicit input value ranges:
  - raw latent/cached feature: real-valued tensors;
  - timestep: `0..49` normalized to `[0, 1]`;
  - PSNR: normalized with `[10, 50]` dB bounds.
- Added a text flow diagram from feature/condition inputs to threshold output.
- Added explicit output value range:
  - prediction `[0, 1]`;
  - current labels `[0.10, 0.80]`.
- Added parameter-count table for `2x2x2`, `2x4x4`, `3x4x4`, `4x4x4`, and condition-only variants.

## Architecture Figure

- Added `reports/assets/adaptive_predictor_architecture.svg`.
- Embedded the figure in `reports/report_adaptive_predictor.md`.
- The figure shows:
  - cached latent feature input `[B,128]` and real-valued range;
  - step input `[B,1]`, raw `0..49`, normalized to `[0,1]`;
  - PSNR input `[B,1]`, raw dB, normalized with `[10,50]`;
  - feature projection `[B,128] -> [B,64]`;
  - condition embedding `[B,2] -> [B,64]`;
  - concat `[B,64]+[B,64] -> [B,128]`;
  - prediction head `[B,128] -> [B,1]`;
  - output threshold range `[0,1]`;
  - current label range `[0.10,0.80]`;
  - default parameter count `29377`.

## Figure Cleanup

- Rewrote the SVG to make it cleaner for PPT use.
- Intermediate blocks now show only:
  - module name;
  - shape transition.
- Input and output blocks retain value ranges.
- Removed detailed layer lists from the visual figure.

## Validation

- Previewed the appended section with `sed`.
- No code, data, or experiment outputs were changed.

## Files Changed

- `reports/report_adaptive_predictor.md`
- `PROGRESS.md`
- `logs/2026-06-16_adaptive_predictor_report_architecture.md`
