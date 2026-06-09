# 2026-06-09 Block-cache-only experiment scripts

- User requested scripts, modeled on previous experiments, to compare two block cache schemes with block cache only: existing `bwcache` and new `block-group`, one prompt each, three thresholds per scheme.
- Added `experiments/block_cache_only_50step_45f_480p/run_experiments.sh`. It runs baseline no-cache, then BWCache thresholds and block-group thresholds with timestep/cfg cache disabled, archiving commands, logs, compute time files, ffprobe JSON, PSNR JSON/logs, failure records, summary outputs, and an `experiment_results/` symlink.
- Added `summarize_results.py` to parse BWCache and block-group cache summaries, produce `results/summary.csv`, and aggregate by method/threshold.
- Added `README.md` documenting defaults, overrides, run command, and output layout.
- Defaults: prompt offset 0, seed 42, T2V-A14B, 832x480, 45 frames, 50 dpm++ steps, BWCache thresholds `0.05 0.15 0.30`, block-group thresholds `0.01 0.03 0.05`, group size 5, pooled relative-L1.
- Validation: `bash -n` passed, conda `py_compile` passed for summarizer and reused helpers, and prompt parser returned prompt 01. No full GPU run launched.
