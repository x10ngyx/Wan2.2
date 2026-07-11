# 2026-07-10 RL4Acc Terminal Speedup Proxy Alignment

## What Changed

- Changed `predicotr-rl/data.py` terminal speedup reward from measured summary speedup to final action-cost proxy speedup.
- Added `compute_final_speedup_proxy()` to compute the final trajectory speedup from reuse/recompute actions and `reuse_cost_ratio`.
- Updated `predicotr-rl/README.md` reward documentation to use `final_proxy_speedup`.

## Rationale

The state includes `Speedup_current`, which is a proxy computed from action counters. Using measured summary speedup only in `R_terminal` made the state progress signal and terminal speedup penalty come from different dynamics. The terminal penalty now uses the same proxy family after all 50 steps.

Measured speedup is still retained in the dataset bundle as `achieved_speedups` for analysis and calibration.

## Validation

- `python -m py_compile predicotr-rl/data.py predicotr-rl/train_iql.py predicotr-rl/policy.py predicotr-rl/models.py` passed.

## Notes

- No Wan2.2 inference or RL training was run.
- `predicotr-rl/` is currently untracked in git, so normal `git diff` does not show file-level changes until the directory is added.
