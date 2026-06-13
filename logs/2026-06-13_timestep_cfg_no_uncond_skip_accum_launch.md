# 2026-06-13 Sea Timestep + Sea CFG No-Skip-Accum

## Summary

- Selected sea timestep + sea CFG behavior: when CFG cache reuses and skips `uncond`, the skipped `uncond` branch does not advance sea timestep accumulated distance.
- The alternative skipped-uncond accumulation implementation was later removed from code and its temporary restore point was deleted.

## Validation

- Ran:
  - `/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m py_compile wan/timestep_cache.py wan/text2video.py wan/cfg_cache.py generate.py experiments/timestep_cfg_prompt01_50step_45f_480p/run_batch.py`
  - `/hy-tmp/miniconda3/envs/Wan2.2/bin/python experiments/timestep_cfg_prompt01_50step_45f_480p/run_batch.py --cpu_validate`
- Confirmed no existing tmux server before launch and A100 was idle.

## Launch

- tmux session: `timestep_cfg_noaccum_p01_20260613_213000`
- Result root: `/hy-tmp/wan22_timestep_cfg_prompt01_no_uncond_skip_accum_50step_45f_480p_20260613_213000`
- Symlink: `experiment_results/wan22_timestep_cfg_prompt01_no_uncond_skip_accum_50step_45f_480p_20260613_213000`
- Candidate grid:
  - sea timestep `0.10`, sea CFG `0.10`
  - sea timestep `0.10`, sea CFG `0.20`
  - sea timestep `0.20`, sea CFG `0.10`
  - sea timestep `0.20`, sea CFG `0.20`
- Launch check found the process running and `pipeline_init.log` started.

## Completion

- Checked on 2026-06-14.
- Runner ended with `Completed experiment`.
- `failed/` is empty.
- All four candidates have complete video, ffprobe, PSNR, log, command, and result artifacts.
- All outputs are `832x480`, `45` frames, `16 fps`, duration `2.8125s`.
- Result table: `/hy-tmp/wan22_timestep_cfg_prompt01_no_uncond_skip_accum_50step_45f_480p_20260613_213000/results/summary.csv`.

## Results

| Sea timestep | Sea CFG | Compute seconds | Speedup | Mean PSNR | Min PSNR |
|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.10 | 489.768 | 1.067x | 36.747 dB | 35.04 dB |
| 0.10 | 0.20 | 415.997 | 1.256x | 26.430 dB | 23.14 dB |
| 0.20 | 0.10 | 348.803 | 1.498x | 24.433 dB | 21.84 dB |
| 0.20 | 0.20 | 337.057 | 1.550x | 24.848 dB | 22.11 dB |

## Comparison To Skip-Accounting Run

| Sea timestep | Sea CFG | Speedup before | Speedup no-accum | PSNR before | PSNR no-accum |
|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.10 | 1.073x | 1.067x | 36.747 dB | 36.747 dB |
| 0.10 | 0.20 | 1.250x | 1.256x | 26.430 dB | 26.430 dB |
| 0.20 | 0.10 | 1.448x | 1.498x | 24.633 dB | 24.433 dB |
| 0.20 | 0.20 | 1.517x | 1.550x | 24.756 dB | 24.848 dB |

## Takeaway

- Disabling skipped-uncond accumulation improved or matched the combined sea timestep + sea CFG tradeoff in this grid.
- The best combined point is `0.20+0.20`: `1.550x`, `24.848 dB`.
- It is close to but still slightly slower than sea timestep-only `0.20` from the prompt-01 SeaCache run (`1.569x`, `24.558 dB`), while giving slightly higher PSNR.
- Keep the no-accum behavior as the maintained sea timestep + sea CFG composition.
