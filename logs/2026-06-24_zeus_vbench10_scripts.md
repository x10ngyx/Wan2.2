# 2026-06-24 ZEUS VBench10 Scripts

- Read `PROGRESS.md` at session start.
- Added `experiments/zeus_vbench10_50step_45f_480p/` for rerunning fixed ZEUS and ZEUS-threshold on `test_sets/Vbench10/prompts.jsonl`.
- Added:
  - `run_batch.py`
  - `summarize_results.py`
  - `run_tmux.sh`
  - `README.md`
- Runner behavior:
  - single-process pipeline load;
  - generates no-cache baselines, fixed ZEUS candidates, and ZEUS-threshold candidates;
  - default ZEUS-threshold sweep is `0.005 0.02 0.08 0.20 0.60`;
  - fixed ZEUS config matches the earlier formal run: acc range `8-47`, denominator `3`, modular `0 1`, `reuse_interp`, max interval `6`, lagrange `4/4/24`;
  - block cache and CFG cache are disabled.
- Validation:
  - `/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m py_compile experiments/zeus_vbench10_50step_45f_480p/run_batch.py experiments/zeus_vbench10_50step_45f_480p/summarize_results.py`
  - `/hy-tmp/miniconda3/envs/Wan2.2/bin/python experiments/zeus_vbench10_50step_45f_480p/run_batch.py --cpu_validate`
  - `/hy-tmp/miniconda3/envs/Wan2.2/bin/python experiments/zeus_vbench10_50step_45f_480p/run_batch.py --cpu_validate --prompt_limit 1 --thresholds '0.005 0.02'`
- No GPU inference, ffprobe, or PSNR jobs were launched.
