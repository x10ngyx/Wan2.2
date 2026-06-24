# ZEUS VBench10 50-step 45f 480p

Runs timestep-cache-only fixed ZEUS and ZEUS-threshold on the unified
`VBench10` prompt subset.

## Configuration

- task: `t2v-A14B`
- checkpoint: `/hy-tmp/models/Wan2.2-T2V-A14B`
- prompts: `test_sets/Vbench10/prompts.jsonl`
- seed: `42`
- size: `832*480`
- frames: `45`
- steps: `50`
- solver: `dpm++`
- timestep cache methods:
  - fixed ZEUS: `--timestep_cache zeus`
  - ZEUS-threshold: `--timestep_cache zeus-threshold`
- block cache: none
- CFG cache: none

The runner reuses existing VBench10 no-cache baselines by default from:

```text
experiment_results/wan22_vbench10_three_cache_full_merge_and_timestep_only_full_20260623
```

It symlinks the baseline video, timing, ffprobe, and log artifacts into the new
experiment root, then loads the WanT2V pipeline once and generates all fixed-ZEUS
and ZEUS-threshold candidates sequentially in the same process. Set
`--baseline_reuse_root ""` when calling `run_batch.py` directly if a fresh
baseline regeneration is required.

## Old ZEUS Settings Preserved

Fixed ZEUS uses the same formal 10-prompt configuration:

- acc range: `8 <= step < 47`
- denominator: `3`
- modular: `0 1`
- caching mode: `reuse_interp`
- max interval: `6`
- lagrange term/int/step: `4 / 4 / 24`

ZEUS-threshold uses the same reuse_interp threshold sweep as the old 10-prompt
reference run:

```text
0.005 0.02 0.08 0.20 0.60
```

## Validate Without GPU

```bash
python experiments/zeus_vbench10_50step_45f_480p/run_batch.py --cpu_validate
```

Expected default work with baseline reuse: `0` baseline generations, `10`
fixed-ZEUS candidates, and `50` ZEUS-threshold candidates.

## Launch

```bash
bash experiments/zeus_vbench10_50step_45f_480p/run_tmux.sh
```

Optional smoke test:

```bash
PROMPT_LIMIT=1 THRESHOLDS="0.005 0.02" \
  EXP_ROOT=/hy-tmp/wan22_zeus_vbench10_smoke \
  SESSION=wan22_zeus_vbench10_smoke \
  bash experiments/zeus_vbench10_50step_45f_480p/run_tmux.sh
```

## Outputs

The default output root is:

```text
/hy-tmp/wan22_zeus_vbench10_50step_45f_480p_<timestamp>
```

Important files:

- `runner.log`: full batch log.
- `baseline/*.mp4`: no-cache baseline videos.
- `zeus/*.mp4`: fixed-ZEUS candidate videos.
- `zeus_threshold/th_<threshold>/*.mp4`: ZEUS-threshold candidate videos.
- `ffprobe/*.json`: generated video metadata.
- `psnr/zeus/*.json`: fixed-ZEUS PSNR results.
- `psnr/zeus_threshold/th_<threshold>/*.json`: ZEUS-threshold PSNR results.
- `results/summary.csv`: one row per candidate.
- `results/aggregate_by_method.csv`: aggregate speed/quality by method and threshold.
- `manifests/selected_vbench10_records.jsonl`: exact prompt records.
