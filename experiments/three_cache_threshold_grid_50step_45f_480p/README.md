# Three-cache threshold grid, 50 steps, 45 frames, 480p

Runs a single-process prompt 01 grid for three enabled caches:

- timestep cache: `zeus-threshold`
- block cache: `block-group`
- CFG cache: `threshold`

The runner creates one `WanT2V` pipeline and sequentially evaluates threshold
combinations in the same process to avoid repeated checkpoint loading.

Default thresholds:

- timestep: `0.001 0.005 0.02 0.60`
- block-group: `0.001 0.015 0.03 1.00`
- CFG: `0.001 0.02 0.03 1.00`

CFG miss forced recomputation is disabled:

```text
force_uncond_recompute_on_miss=False
```

Launch:

```bash
bash experiments/three_cache_threshold_grid_50step_45f_480p/run_tmux.sh
```

Override thresholds:

```bash
TIMESTEP_THRESHOLDS="0.001 0.005 0.02 0.60" \
BLOCK_THRESHOLDS="0.001 0.015 0.03 1.00" \
CFG_THRESHOLDS="0.001 0.02 0.03 1.00" \
bash experiments/three_cache_threshold_grid_50step_45f_480p/run_tmux.sh
```
