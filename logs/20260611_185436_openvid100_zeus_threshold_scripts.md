# 2026-06-11 OpenVid-100 ZEUS-Threshold Scripts

- Added `experiments/zeus_threshold_openvid100_50step_45f_480p/`.
- New runner reads `/hy-tmp/openvid_100_wan22_prompts.zip` directly and uses the `dataset_100.jsonl` records as the prompt manifest.
- Default experiment uses only timestep cache with `zeus-threshold`; block cache and CFG cache are disabled.
- Generation/evaluation settings are aligned with earlier Wan2.2 threshold experiments: fixed seed 42, `832*480`, 45 frames, 50 dpm++ steps, `offload_model=True`, `convert_model_dtype=True`, `reuse_interp`, `acc_range=(8,47)`, `max_interval=6`, FFmpeg PSNR against same-prompt no-cache baseline.
- Default thresholds: `0.001 0.003 0.005 0.008 0.010 0.015 0.020 0.030 0.050 0.080`.
- Full default workload is 100 no-cache baselines and 1000 ZEUS-threshold candidate generations. One WanT2V pipeline is loaded per process and reused for all selected runs.
- Archive output includes videos, command records, logs, compute-only time files, ffprobe JSON, PSNR JSON/logs, selected manifests, failure records, per-row summary, threshold aggregates, and threshold-by-group aggregates.
- CPU validation passed: `py_compile`, `bash -n`, `run_batch.py --help`, `run_batch.py --cpu_validate`, and a temporary summarizer fixture.
- No GPU run was launched.
