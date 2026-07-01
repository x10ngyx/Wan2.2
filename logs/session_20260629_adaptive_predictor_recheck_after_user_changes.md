# 2026-06-29 Adaptive Predictor Recheck After User Changes

## Scope

- Rechecked the user-updated adaptive predictor files after the Conv3d patch revision.
- Focused on `adaptive_threshold_predictor/models.py`, `train_gate.py`, `data.py`, `build_grid_feature_cache.py`, and updated README/report notes.

## Result

- The new `MiniDiTCLSAdaptiveThresholdPredictor` now consumes raw latent `[B,16,12,60,104]` and uses a learnable `Conv3d` patch embedding with patch size `(3,12,8)`.
- `mini_dit_cls` no longer requires `grid_cache_dir`; it uses `TraceStepThresholdDataset`.
- Checkpoint/metrics metadata now includes `feature_extractor` details needed for online reconstruction.
- Transformer training defaults now match the architecture report better: batch size 64, epochs 30, lr `3e-4`, SmoothL1 beta `0.02`, grad clip `1.0`, early stopping 5.

## Validation

- Ran `py_compile` on adaptive predictor modules: passed.
- Ran a reduced MiniDiT forward pass: output shape `[1,1]`, threshold within `[0.10,0.80]`.
- Instantiated default MiniDiT config: `724,513` trainable parameters, `Conv3d(16,96,kernel=(3,12,8),stride=(3,12,8))`, grid `(16,4,5,13)`.
- `git diff --check` on adaptive predictor files: passed.

## Remaining Notes

- Main unresolved risk is still dataset semantics, not architecture mechanics: `candidate_inverse` trains with achieved PSNR and fixed-threshold candidate latents, while deployment needs desired PSNR and adaptive-run latents.
- Shared `train_gate.py` defaults now favor MiniDiT; old MLP reproduction commands should explicitly set legacy hyperparameters if exact comparison is needed.
- No code changes were made besides this log and the `PROGRESS.md` entry.
