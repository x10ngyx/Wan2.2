# 2026-07-11 Predictor-RL Speedup-Priority Defaults

## Change

- Updated `predicotr-rl/data.py` and `predicotr-rl/train_iql.py` default `lambda_speedup` from `10.0` to `30.0`.
- Updated `predicotr-rl/train_iql.py` default IQL `beta` from `3.0` to `1.5`.
- Updated `predicotr-rl/README.md` reward-default documentation and rationale.

## Rationale

- The preceding full OpenVid-100 summary analysis found that the median break-even speedup coefficient for the low-speed `threshold 0.10 -> 0.15` transition is `30.13`, and the next transition requires `17.20`. A default of `30.0` prioritizes speed-target adherence through the low/mid speed range rather than preserving the former high-PSNR low-speed trajectory.
- At the new default, a `0.15x` speed mismatch costs `4.5` terminal reward units and a `0.30x` mismatch costs `9.0`. This is intentionally stronger than the former `1.5` and `3.0` penalties.
- The training loop standardizes advantages before using `exp(beta * advantage)`. Lowering beta to `1.5` reduces behavior-cloning weight concentration while keeping the stronger speed-conditioned Q/value ranking responsible for speed-target control.

## Scope and Validation

- Retained latent, recompute, PSNR, speed-proxy, expectile, discount, target-Q, model-capacity, and optimizer defaults.
- No checkpoint was regenerated and no Wan2.2 inference code was changed.
- GPU training and closed-loop validation remain pending because no NVIDIA GPU is visible in this session.
