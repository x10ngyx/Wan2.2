# ZEUS Threshold Timestep Cache T2V Experiment

Configuration:

- Model: Wan2.2 T2V-A14B
- Checkpoint: `/hy-tmp/models/Wan2.2-T2V-A14B`
- Prompts: `/hy-tmp/work/Wan2.2/prompt.txt` (10 prompts)
- Steps: 50
- Frames: 45
- Resolution: 480p landscape (`832*480`)
- Solver: `dpm++` by default, override with `SAMPLE_SOLVER=unipc`
- Methods: baseline (`--timestep_cache none`) and `zeus-threshold`
- Default thresholds: `0.03 0.08 0.15 0.30 0.60`

Run:

```bash
bash experiments/zeus_threshold_50step_45f_480p/run_experiments.sh
```

Override thresholds:

```bash
THRESHOLDS="0.02 0.05 0.10 0.20 0.40" \
bash experiments/zeus_threshold_50step_45f_480p/run_experiments.sh
```

The script writes all outputs under `/hy-tmp`, including:

- generated videos for baseline and every threshold
- full command records
- raw run logs and timing files
- ffprobe JSON for each video
- PSNR JSON/logs against the same prompt/seed baseline
- per-prompt CSV summary
- per-threshold aggregate CSV/JSON
- failure records under `failed/`

## Implementation Notes

This experiment targets the new `--timestep_cache zeus-threshold` path. It keeps the ZEUS output reuse modes and `(model_stage, branch)` state separation, but replaces the fixed ZEUS skip cadence with direct input-latent relative-L1 thresholding.

Block cache and CFG cache remain disabled through:

```bash
--block_cache none --cfg_cache none
```
