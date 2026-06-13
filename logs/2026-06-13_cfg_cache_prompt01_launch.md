# 2026-06-13 CFG Cache Prompt-01 Launch And Results

## Scope

Started a cfg-cache-only prompt-01 comparison in tmux to compare the original CFG threshold method against the new SeaCache-aligned CFG cache.

## Implementation

Added:

- `experiments/cfg_cache_prompt01_50step_45f_480p/run_batch.py`
- `experiments/cfg_cache_prompt01_50step_45f_480p/run_tmux.sh`

The runner loads one WanT2V pipeline and runs all CFG-only candidates sequentially. It reuses the previous prompt-01 baseline video and baseline compute time.

## Experiment

- tmux session: `cfg_cache_p01_20260613_163243`
- result root: `/hy-tmp/wan22_cfg_cache_prompt01_50step_45f_480p_20260613_163243`
- symlink: `experiment_results/wan22_cfg_cache_prompt01_50step_45f_480p_20260613_163243`
- baseline root: `/hy-tmp/wan22_cache_ablation_prompt01_50step_45f_480p_20260609_184625`
- baseline compute seconds: `522.603`
- prompt: prompt-01 from `prompt.txt`
- seed: `42`
- size: `832*480`
- frames: `45`
- sample steps: `50`
- solver: `dpm++`
- offload: enabled
- dtype conversion: enabled
- timestep cache: disabled
- block cache: disabled

Candidates:

- `threshold:0.02`
- `threshold:0.03`
- `sea-threshold:0.10`
- `sea-threshold:0.20`
- `sea-threshold:0.30`

## Validation Before Launch

Ran:

```bash
/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m py_compile experiments/cfg_cache_prompt01_50step_45f_480p/run_batch.py wan/cfg_cache.py wan/text2video.py generate.py
/hy-tmp/miniconda3/envs/Wan2.2/bin/python experiments/cfg_cache_prompt01_50step_45f_480p/run_batch.py --cpu_validate
bash -n experiments/cfg_cache_prompt01_50step_45f_480p/run_tmux.sh
```

GPU check before launch showed A100 80GB available and idle.

## Launch Check

After launch:

- tmux session was active.
- `runner.log` showed checkpoint loading completed.
- first candidate `threshold_th_0p02` had entered 50-step sampling.
- `nvidia-smi` showed the GPU under active load.

## Completion

The tmux session exited after completing all five candidates. No files were present under `failed/`.

All candidates have:

- video output,
- ffprobe JSON,
- PSNR JSON,
- candidate log,
- command record.

Main result files:

- `/hy-tmp/wan22_cfg_cache_prompt01_50step_45f_480p_20260613_163243/results/summary.csv`
- `/hy-tmp/wan22_cfg_cache_prompt01_50step_45f_480p_20260613_163243/results/summary_with_cache.csv`
- `/hy-tmp/wan22_cfg_cache_prompt01_50step_45f_480p_20260613_163243/results/summary.json`

## Results

| candidate | speedup | mean PSNR | min PSNR | CFG reuse/recompute |
|---|---:|---:|---:|---:|
| `threshold_th_0p02` | 1.041x | 26.732 dB | 22.89 dB | 9/41 |
| `threshold_th_0p03` | 1.137x | 21.571 dB | 20.31 dB | 17/33 |
| `sea_threshold_th_0p10` | 1.007x | 37.457 dB | 34.81 dB | 6/44 |
| `sea_threshold_th_0p20` | 1.175x | 26.226 dB | 23.13 dB | 20/30 |
| `sea_threshold_th_0p30` | 1.297x | 21.359 dB | 20.07 dB | 28/22 |

## Takeaway

On prompt-01, the new Sea-style CFG cache gives a better speed/quality tradeoff than the old threshold method at comparable quality levels. `sea-threshold 0.20` is the most useful initial point: it is faster than old `threshold 0.03` while keeping PSNR close to old `threshold 0.02`. `sea-threshold 0.10` is very conservative and high quality, while `sea-threshold 0.30` is the aggressive speed point.
