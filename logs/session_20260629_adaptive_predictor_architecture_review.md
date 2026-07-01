# 2026-06-29 Adaptive Predictor Architecture Review

## Scope

- Reviewed the original MLP predictor in `adaptive_threshold_predictor/models.py`, `data.py`, `train_gate.py`, `build_feature_cache.py`, and prior reports.
- Reviewed the new Transformer candidate in `MiniDiTCLSAdaptiveThresholdPredictor`, `GridFeatureThresholdDataset`, `build_grid_feature_cache.py`, and the smoke outputs under `/hy-tmp/wan22_mini_dit_*_20260629`.

## Findings

- Original MLP:
  - Uses pooled latent-derived features plus timestep/PSNR condition branch.
  - Main formal data path is `candidate_inverse`: candidate latent + achieved PSNR -> threshold.
  - Historical best lightweight setting was `2x2x2 temporal_mean`; `hidden_dim=16` matched hdim64 validation loss with far fewer parameters.
- New Transformer:
  - Uses fixed average-pooled grid features `[16,4,5,13]`, factorized learned 3D position embeddings, CLS readout, AdaLN-style conditioning, and threshold range `[0.10,0.80]`.
  - Differs from the architecture proposal by using `avg_pool3d` plus `Linear(16->dim)` instead of a learnable `Conv3d` patch embedding.
  - Smoke artifacts show CPU smoke ran on only 16 examples from one sample, so validation split was empty and not meaningful.
- Main risks:
  - `candidate_inverse` has a deployment mismatch because online inference supplies desired PSNR, not achieved PSNR, and adaptive-run latent distribution differs from fixed-threshold candidate latents.
  - `train_gate.py` defaults remain legacy-short-run defaults; Transformer runs require explicit overrides for lr, batch size, loss beta, grad clipping, and early stopping.
  - Online reconstruction needs the grid feature extraction config, not only the model state dict.

## Validation

- Ran a lightweight Python forward check for MLP raw/cached paths and `grid_mlp`; outputs had expected `[B,1]` shape.
- MiniDiT CPU forward on full grid was interrupted due slow CPU execution; existing smoke artifacts already confirm construction and training entry path, but not meaningful validation quality.

## Code Changes

- No predictor code was changed.
- Updated `PROGRESS.md` and added this session log.
