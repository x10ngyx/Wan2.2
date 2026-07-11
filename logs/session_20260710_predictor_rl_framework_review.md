# 2026-07-10 Predictor-RL Framework Review

## Scope

- Reviewed `/hy-tmp/work/Wan2.2/predicotr-rl` for method, code implementation, and default parameter reasonableness.
- Read `PROGRESS.md`, `predicotr-rl/README.md`, `data.py`, `train_iql.py`, `models.py`, and `policy.py`.
- Checked OpenVid-100 `summary.csv` ranges for PSNR, speedup, and reuse-step counts.

## Findings

- The framework is best described as an offline-IQL prototype over fixed SeaCache threshold trajectories, not a validated online RL predictor.
- Main method risks are limited counterfactual action coverage, distribution shift when the learned policy composes new skip/recompute paths, and no closed-loop Wan2.2 runtime evaluation yet.
- Main parameter risk is reward scaling: raw PSNR spans about `13.6-48.9 dB`, while default local target-speedup offsets are only `±0.3` with `lambda_speedup=1`, so the target-speedup condition can be weak relative to PSNR.
- Main implementation risks are selecting `best_model.pt` by validation behavior-action accuracy, batch-normalizing advantages before exponentiation, and expensive bundle construction when latent-MSE/state preprocessing is not cached.

## Changes

- No predictor, training, inference, cache, or experiment code was changed.
- Added this session log and a concise `PROGRESS.md` note.
