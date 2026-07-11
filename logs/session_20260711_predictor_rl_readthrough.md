# 2026-07-11 Predictor-RL Readthrough

## Scope

- Read `PROGRESS.md`, the `predicotr-rl/` README, data builder, IQL models/trainer, policy loader, and previous predictor-RL session records.
- Checked for IQL training artifacts and GPU availability.

## Findings

- `predicotr-rl/` implements an offline IQL prototype over OpenVid-100 fixed SeaCache trajectories, not an online RL environment or an already-integrated Wan2.2 inference path.
- Each denoising timestep has one synchronized action for cond/uncond branches. `1` is reuse/skip and `0` is recompute.
- State is the five cached latent feature sets plus timestep, requested speedup, projected full-task speedup, and consecutive-skip count. The terminal speedup objective uses the same calibrated action-cost proxy as the state.
- The immediate reward penalizes latent drift and recompute; the terminal reward combines video PSNR with absolute proxy-speedup target error. Training duplicates each historical trajectory over local speedup-target offsets.
- The trainer saves the best checkpoint by validation behavior-action accuracy. This is a diagnostic of imitation on held-out traces, not a closed-loop quality/speed selection metric.
- Available `/hy-tmp/wan22_iql_*` checkpoints are CPU smoke-test outputs. No full latent-MSE cache, full GPU training run, policy-to-SeaCache adapter, or archived Wan2.2 closed-loop evaluation was found.
- `nvidia-smi` reported no visible device during this session; no GPU activity was attempted.

## Changes and Validation

- No implementation, cache logic, checkpoints, datasets, or experiments were changed.
- Updated `PROGRESS.md` with this readthrough summary.
