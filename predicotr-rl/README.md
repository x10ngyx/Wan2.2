# Predictor RL

Offline IQL prototype for the adaptive SeaCache predictor.

This directory is intentionally separate from the existing
`adaptive_threshold_predictor/` code.  The first implementation keeps the same
adaptive SeaCache goal as the earlier predictor work, but replaces the
threshold-regression network with an offline RL policy trained from SeaCache
trace data.

Scope for this prototype:

- timestep-only adaptive SeaCache decisions;
- one synchronized skip/recompute action per denoising step;
- cond/uncond branches use the same per-step decision;
- state uses the five existing latent feature sets:
  `latent_pool`, `temporal_mean`, `temporal_var`, `frame_diff_mean`,
  `frame_diff_var`;
- offline IQL with independent V, Q1, Q2, and policy MLPs.

The directory name follows the user-requested spelling `predicotr-rl`.

## Training

The trainer consumes the existing feature cache:

```bash
python predicotr-rl/train_iql.py \
  --feature_cache /hy-tmp/wan22_adaptive_threshold_feature_cache_candidate_inverse_20260616_012409 \
  --data_root /hy-tmp/openvid_100_seacache_trace_data \
  --out_dir /hy-tmp/wan22_iql_seacache_debug \
  --target_speedup_offsets -0.3 -0.15 0.0 0.15 0.3 \
  --device cuda
```

Use `--max_examples` and `--device cpu` for a fast static smoke test.

Before full training, prepare the latent-MSE reward cache:

```bash
python predicotr-rl/prepare_data.py \
  --feature_cache /hy-tmp/wan22_adaptive_threshold_feature_cache_candidate_inverse_20260616_012409 \
  --data_root /hy-tmp/openvid_100_seacache_trace_data
```

This writes `latent_mse_to_baseline.pt` and `latent_mse_to_baseline.json` under
the feature-cache root.  The training script uses that cache by default.

Reward follows the PDF formulas:

```text
r_t = -lambda_latent * MSE(z_{t-1}, z_gt_{t-1})
      -lambda_recompute * (1 - action)

R_terminal = lambda_psnr * PSNR(Video_pred, Video_gt)
             -lambda_speedup * abs(final_proxy_speedup - target_speedup)
```

Default reward weights are calibrated to keep the terms on comparable scales:

```text
lambda_latent = 5.0
lambda_recompute = 0.04
lambda_psnr = 1.0
lambda_speedup = 30.0
```

The defaults prioritize target-speed adherence.  On OpenVid-100, the
`0.10 -> 0.15` SeaCache threshold transition has a median break-even speedup
weight of about `30`, because its quality loss is steep relative to its `0.3x`
speed gain.  `lambda_speedup=30.0` therefore makes a `0.3x` target miss cost
`9` reward units, while retaining the raw PSNR quality objective.  The IQL
default uses `beta=1.5` with batch-normalized advantages to avoid the highly
concentrated policy weights produced by the earlier `beta=3.0` setting.

`target_speedup_offsets` duplicates each offline SeaCache trajectory across
nearby requested speed targets around its measured speedup.  For example, a
trajectory with measured `2.4x` speedup gets targets `2.1`, `2.25`, `2.4`,
`2.55`, and `2.7` with the default offsets.  This keeps the target condition
near the behavior trajectory while making the terminal speedup penalty nonzero.
The per-step latent MSE is read from `baseline_step_inputs` and
`seacache_step_inputs` and is cached by default as `latent_mse_to_baseline.pt`
under the feature-cache root.

`Speedup_current` is computed in the full-task sense: completed steps use the
observed reuse/recompute cost proxy, while all unfinished steps are temporarily
counted as full recompute cost.  The terminal speedup penalty uses the same
action-cost proxy after all 50 steps, so the state progress signal and terminal
target-speedup reward are defined by the same dynamics.  The measured summary
speedup is still stored for analysis.  The default `reuse_cost_ratio=0.081` is
calibrated from OpenVid-100 SeaCache traces; it reduces final-speedup proxy
error from about 5% MAPE at `0.05` to about 0.32% MAPE on that table.

The checkpoint stores:

- V/Q/policy network weights;
- target Q weights;
- state normalization statistics;
- training config and feature metadata.

Runtime integration is deliberately not implemented here yet.  The next step is
to add a thin SeaCache policy adapter that loads the exported policy and calls it
where the current adaptive predictor calls its threshold network.

`policy.py` provides that thin loader without modifying Wan2.2 runtime code:

```python
from pathlib import Path
import sys

sys.path.insert(0, "predicotr-rl")
from policy import SeaCacheRLPolicy

policy = SeaCacheRLPolicy(Path("/hy-tmp/wan22_iql_seacache_debug/best_model.pt"))
decision = policy.decide(
    features=five_feature_dict,
    step_index=step_index,
    num_steps=50,
    target_speedup=2.0,
    reuse_count=reuse_count,
    recompute_count=recompute_count,
    consecutive_skip=consecutive_skip,
)
```

`decision["action"] == 1` means reuse/skip; `0` means recompute.
