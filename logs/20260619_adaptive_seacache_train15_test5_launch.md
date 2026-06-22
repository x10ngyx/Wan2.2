# 2026-06-19 Adaptive SeaCache Train15 Test5 Launch

## Summary

- Stopped tmux session `adaptive_seacache_train10` after the user requested a larger run.
- The old root `/hy-tmp/wan22_adaptive_seacache_train10_50step_45f_480p_20260619_134522` is an interrupted pilot; it had completed only one candidate.
- Created `experiments/adaptive_seacache_train15_test5_50step_45f_480p/`.
- The new runner samples 15 prompts from predictor `train_sample_ids` and 5 prompts from `val_sample_ids`, using random seed `20260619`.
- It reuses existing no-cache OpenVid baselines from `/hy-tmp/work/Wan2.2/experiment_results/openvid_100_seacache_trace_data`; no baseline generation is performed.
- It runs timestep-only adaptive SeaCache with target PSNR values `20`, `25`, and `30`, for 60 total candidates.

## Selected Source IDs

Train:

```text
openvidhd_part1_085
openvidhd_part1_086
openvidhd_part1_059
openvidhd_part1_057
openvidhd_part1_016
openvidhd_part1_036
openvidhd_part1_093
openvidhd_part1_063
openvidhd_part1_095
openvidhd_part1_058
openvidhd_part1_027
openvidhd_part1_012
openvidhd_part1_020
openvidhd_part1_031
openvidhd_part1_037
```

Test/val:

```text
openvidhd_part1_028
openvidhd_part1_030
openvidhd_part1_026
openvidhd_part1_092
openvidhd_part1_055
```

## Validation

- `py_compile` passed for `run_batch.py`.
- `bash -n` passed for `run_tmux.sh`.
- CPU validation passed and found all baseline video/log/time/ffprobe artifacts.
- GPU check before launch: `NVIDIA A100 80GB PCIe`, 0 MiB used.

## Launch

- tmux session: `adaptive_seacache_train15_test5`
- experiment root: `/hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521`
- runner log: `/hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/logs/runner.log`
- Initial log check showed the adaptive gate checkpoint loaded successfully.

## Notes

- No git commit was made.
- The experiment is still running at session handoff time.
