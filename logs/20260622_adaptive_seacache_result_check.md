# 2026-06-22 Adaptive SeaCache Result Check

## Actions

- Created symlinks under `experiment_results/` for:
  - `/hy-tmp/wan22_adaptive_seacache_train10_50step_45f_480p_20260619_134522`
  - `/hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521`
  - `/hy-tmp/wan22_adaptive_seacache_overhead_train5_50step_45f_480p_20260619_143632`
- Checked tmux: no sessions were running.
- Checked completion counts, failed files, runner logs, ffprobe JSONs, PSNR JSONs, traces, and time files.
- Reconstructed partial summaries from existing artifacts because no run reached normal final-summary writing.

## Status

- `train10`: interrupted pilot, 1 candidate completed, no failed file.
- `train15_test5`: 12 candidates completed, one OOM failure at `openvidhd_part1_016 target=20`.
- `overhead_train5`: 22 rows completed, one OOM failure at `openvidhd_part1_057 target=30`.
- Existing ffprobe outputs all matched `832x480`, `45` frames.

## Generated Tables

- `/hy-tmp/wan22_adaptive_seacache_train10_50step_45f_480p_20260619_134522/results/partial_summary.csv`
- `/hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/results/partial_summary.csv`
- `/hy-tmp/wan22_adaptive_seacache_overhead_train5_50step_45f_480p_20260619_143632/results/partial_summary.csv`

## Key Findings

- `train15_test5` completed only the first four train prompts for all three targets.
- On those four prompts, target 20/25/30 showed increasing mean PSNR and decreasing speedup overall, but per-prompt monotonicity is not guaranteed.
- Overhead experiment completed 11 online/replay pairs with zero decision mismatches.
- Predictor timing overhead was about `0.195s` per candidate on average, roughly `0.096%` of online compute time.
- Replay-based overhead was about `0.218s` per candidate on average, roughly `0.12%` of online compute time.

## Notes

- No retry was launched during this check.
- No git commit was made.
