# SeaCache vs ZEUS-threshold on Prompts 01-02

## Shared Settings

- Model/task: Wan2.2 `t2v-A14B`
- Checkpoint: `/hy-tmp/models/Wan2.2-T2V-A14B`
- Shape: `832*480`, 45 frames, 50 DPM++ steps
- Seed: `42`
- Timing: compute-only inference time
- PSNR: FFmpeg PSNR vs same prompt/seed no-cache baseline, perfect frames excluded
- Baseline times:
  - Prompt 01: `522.603s`
  - Prompt 02: `522.608s`

## Prompt 01

| Method | Threshold | Time (s) | Speedup | PSNR (dB) | Reuse / Recompute |
|---|---:|---:|---:|---:|---:|
| ZEUS-threshold | 0.005 | 470.287 | 1.111x | 26.954 | 5 / 45 |
| ZEUS-threshold | 0.020 | 325.253 | 1.607x | 18.606 | 19 / 31 |
| ZEUS-threshold | 0.080 | 232.145 | 2.251x | 18.873 | 28 / 22 |
| ZEUS-threshold | 0.200 | 201.093 | 2.599x | 18.900 | 31 / 19 |
| ZEUS-threshold | 0.600 | 191.124 | 2.734x | 18.928 | 32 / 18 |
| SeaCache | 0.050 | 529.025 | 0.988x | Infinity | 0 / 50 |
| SeaCache | 0.100 | 469.995 | 1.112x | 36.303 | 6 / 44 |
| SeaCache | 0.200 | 333.084 | 1.569x | 24.558 | 20 / 30 |
| SeaCache | 0.300 | 265.758 | 1.966x | 20.562 | 27 / 23 |
| SeaCache | 0.500 | 188.053 | 2.779x | 19.460 | 35 / 15 |

## Prompt 02

| Method | Threshold | Time (s) | Speedup | PSNR (dB) | Reuse / Recompute |
|---|---:|---:|---:|---:|---:|
| ZEUS-threshold | 0.005 | 478.997 | 1.091x | 24.433 | 4 / 46 |
| ZEUS-threshold | 0.020 | 344.151 | 1.519x | 20.112 | 17 / 33 |
| ZEUS-threshold | 0.080 | 230.705 | 2.265x | 20.433 | 28 / 22 |
| ZEUS-threshold | 0.200 | 199.380 | 2.621x | 20.432 | 31 / 19 |
| ZEUS-threshold | 0.600 | 188.933 | 2.766x | 20.438 | 32 / 18 |
| SeaCache | 0.080 | 528.725 | 0.988x | Infinity | 0 / 50 |
| SeaCache | 0.100 | 479.491 | 1.090x | 45.532 | 5 / 45 |
| SeaCache | 0.120 | 421.440 | 1.240x | 42.475 | 11 / 39 |
| SeaCache | 0.150 | 372.696 | 1.402x | 35.441 | 16 / 34 |
| SeaCache | 0.180 | 343.774 | 1.520x | 29.848 | 19 / 31 |
| SeaCache | 0.200 | 334.663 | 1.562x | 30.097 | 20 / 30 |
| SeaCache | 0.250 | 285.500 | 1.831x | 29.055 | 25 / 25 |
| SeaCache | 0.300 | 265.988 | 1.965x | 29.582 | 27 / 23 |
| SeaCache | 0.400 | 217.301 | 2.405x | 27.044 | 32 / 18 |
| SeaCache | 0.500 | 197.847 | 2.641x | 23.725 | 34 / 16 |

## Takeaways

- SeaCache gives a much better quality/speed frontier than ZEUS-threshold on both prompts when quality matters.
- On prompt 01, SeaCache `0.20` gives almost the same speed as ZEUS-threshold `0.02` (`1.569x` vs `1.607x`) but PSNR is `+5.952 dB` higher (`24.558` vs `18.606`). At the aggressive end, SeaCache `0.50` is slightly faster than ZEUS-threshold `0.60` (`2.779x` vs `2.734x`) and still slightly higher PSNR (`19.460` vs `18.928`).
- On prompt 02, SeaCache is clearly stronger. Around `1.5x`, SeaCache `0.18/0.20` gives about `29.85-30.10 dB`, while ZEUS-threshold `0.02` gives `20.112 dB`. Around `2.4x`, SeaCache `0.40` gives `27.044 dB`, while ZEUS-threshold near that speed is about `20.43 dB`.
- ZEUS-threshold PSNR saturates at a low value once reuse becomes aggressive: prompt 01 stays near `18.6-18.9 dB` from `1.6x` to `2.7x`, and prompt 02 stays near `20.1-20.4 dB` from `1.5x` to `2.8x`.
- SeaCache thresholds are not directly numerically comparable with ZEUS-threshold thresholds. The useful SeaCache range in these two prompts is roughly `0.10-0.40`, while ZEUS-threshold only has one acceptable high-quality point at very low reuse (`0.005`) and then quality drops sharply.

## Suggested Next Candidates

- Conservative/high quality: SeaCache `0.12` and `0.15`.
- Balanced: SeaCache `0.20` and `0.30`.
- Aggressive but still plausible: SeaCache `0.40`.

For the next multi-prompt validation, use SeaCache thresholds `0.12 0.15 0.20 0.30 0.40`. Keep ZEUS-threshold `0.005 0.02 0.08 0.20` as comparison baselines if rerunning a matched subset is needed.
