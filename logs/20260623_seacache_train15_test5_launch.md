# 2026-06-23 Fixed SeaCache Train15/Test5 Launch

## What Changed

- Added a fixed-threshold timestep-only SeaCache control runner for the same 20
  OpenVid prompts used by the adaptive SeaCache train15/test5 report.
- New files:
  - `experiments/seacache_train15_test5_50step_45f_480p/run_batch.py`
  - `experiments/seacache_train15_test5_50step_45f_480p/run_tmux.sh`
  - `experiments/seacache_train15_test5_50step_45f_480p/README.md`
- Updated `PROGRESS.md` with the launch state.
- Added workspace result symlink:
  - `experiment_results/wan22_seacache_train15_test5_50step_45f_480p_20260623_160513`

## Experiment

- Thresholds: `0.1 0.2 0.4 0.6`
- Prompt selection: same split JSON and random seed as adaptive train15/test5
  (`20260619`), producing 15 train prompts and 5 held-out test prompts.
- Baselines are reused from
  `/hy-tmp/work/Wan2.2/experiment_results/openvid_100_seacache_trace_data`.
- Result root:
  `/hy-tmp/wan22_seacache_train15_test5_50step_45f_480p_20260623_160513`
- tmux session:
  `seacache_train15_test5`

## Validation

- Confirmed GPU mode before launch:
  `NVIDIA A100 80GB PCIe`, 81920 MiB, driver `570.211.01`.
- Passed:
  - `python -m py_compile experiments/seacache_train15_test5_50step_45f_480p/run_batch.py`
  - `python experiments/seacache_train15_test5_50step_45f_480p/run_batch.py --cpu_validate --thresholds '0.1 0.2 0.4 0.6'`
  - `bash -n experiments/seacache_train15_test5_50step_45f_480p/run_tmux.sh`
- Initial runtime check showed the first candidate had started sampling:
  `openvidhd_part1_085`, threshold `0.1`; GPU was around 47.6 GiB used and
  100% utilization. No failed files were present at that check.

## Follow-Up

- Monitor the tmux session until completion.
- After completion, inspect:
  - `results/summary.csv`
  - `results/aggregate_by_threshold.csv`
  - `failed/`
- Compare the fixed SeaCache aggregate frontier against the adaptive
  target-20/25/30 results in
  `reports/report_adaptive_seacache_train15_test5_and_overhead.md`.

## Completion Check

Checked on 2026-06-24. The fixed SeaCache train15/test5 run completed:

- candidate videos: `80/80`
- PSNR JSON files: `80/80`
- candidate ffprobe JSON files: `80/80`
- command records: `80/80`
- failed files: `0`
- result tables:
  - `/hy-tmp/wan22_seacache_train15_test5_50step_45f_480p_20260623_160513/results/summary.csv`
  - `/hy-tmp/wan22_seacache_train15_test5_50step_45f_480p_20260623_160513/results/aggregate_by_threshold.csv`

Aggregate across all 20 prompts:

| threshold | speedup | mean PSNR | min PSNR |
| --- | ---: | ---: | ---: |
| 0.1 | 1.138x | 42.861 dB | 34.33 dB |
| 0.2 | 1.607x | 30.548 dB | 16.88 dB |
| 0.4 | 2.467x | 23.936 dB | 15.93 dB |
| 0.6 | 3.176x | 21.229 dB | 15.04 dB |

Direct comparison points against adaptive train15/test5:

- Fixed SeaCache `0.6`: `3.176x`, `21.229 dB`; adaptive target-20:
  `3.171x`, `21.108 dB`.
- Fixed SeaCache `0.4`: `2.467x`, `23.936 dB`; adaptive target-25:
  `2.461x`, `23.407 dB`.
- Fixed SeaCache `0.2`: `1.607x`, `30.548 dB`; adaptive target-30:
  `1.904x`, `27.221 dB`.
