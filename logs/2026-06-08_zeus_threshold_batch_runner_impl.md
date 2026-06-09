# 2026-06-08 ZEUS-threshold single-process batch runner implementation

- Added experiments/zeus_threshold_50step_45f_480p/run_batch.py.
- The runner creates one WanT2V pipeline per process, then sequentially runs baseline and threshold generations across selected prompts without reloading checkpoints for every method.
- It preserves the existing experiment archive layout: baseline videos, zeus_threshold videos by threshold label, command records, per-run logs, timing files, ffprobe JSON, PSNR JSON/logs, failed records, summary.csv, and aggregate_by_threshold outputs.
- It supports prompt_start, prompt_limit, thresholds, resume_existing, explicit ffprobe_bin, fixed seed, and the same T2V/threshold settings as the shell script.
- Validation passed: conda py_compile and --help. Full generation smoke was not launched because the existing shell-based prompt 01 pilot is still running and had already reached the final threshold.
