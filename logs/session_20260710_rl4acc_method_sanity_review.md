# 2026-07-10 RL4Acc Method Sanity Review

## What I Checked

- Read `PROGRESS.md` first per project workflow.
- Extracted text from `doc/RL4Acc.pdf` with `pypdf`.
- Compared the PDF method against:
  - `predicotr-rl/README.md`
  - `predicotr-rl/data.py`
  - `predicotr-rl/train_iql.py`
  - `predicotr-rl/models.py`
  - `predicotr-rl/policy.py`

## Conclusion

The current code matches the PDF's high-level offline IQL formulas: binary skip action, state with timestep/features/target speed/current speed/consecutive skip, expectile value update, double-Q Bellman update, advantage-weighted policy update, latent-MSE immediate reward, recompute penalty, and terminal PSNR/speedup reward.

It is not a strict executable reproduction of an online denoising environment. The implementation uses precomputed fixed-threshold SeaCache traces, reconstructs behavior-policy transitions from those rows, duplicates trajectories over a target-speedup grid, uses a calibrated speedup proxy, and currently has no integrated Wan2.2 closed-loop evaluation for the learned policy.

## Files Changed

- Updated `PROGRESS.md` with the review summary.
- Added this session log.

## Validation

- No code was changed.
- No training or inference was run.

## Follow-Up Risks

- Fixed-threshold trace data may not provide enough diverse action coverage for offline RL to compose better policies.
- Validation policy loss is not enough to prove target-speedup/PSNR control; the policy needs closed-loop rollout evaluation.
- Reward scales should be logged and tuned so terminal PSNR does not dominate all step-level rewards.
