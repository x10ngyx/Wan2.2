# 2026-06-12 Reports and Data-Generation Context

- Read `PROGRESS.md` at session start as required.
- Moved all top-level report markdown files into `reports/`:
  - `reports/report.md`
  - `reports/report_main_experiments.md`
  - `reports/report_supplementary_experiments.md`
  - `reports/report_seacache_vs_zeus_threshold_prompt12.md`
- Verified no non-log/non-progress files reference the old top-level report paths.
- Read recent OpenVid and handoff logs:
  - `logs/20260611_184343_openvid_prompt_comparison.md`
  - `logs/20260611_185436_openvid100_zeus_threshold_scripts.md`
  - `logs/20260611_openvid_first50_handoff_package.md`
  - `logs/20260612_000218_seacache_prompt02_highthr_launch.md`
- Checked current SeaCache OpenVid scripts and handoff README. The active data-generation direction is OpenVid-100 SeaCache prompt-threshold data:
  - local scripts: `experiments/seacache_openvid100_50step_45f_480p/`
  - thresholds: `0.08 0.10 0.12 0.15 0.18 0.20 0.25 0.30 0.40 0.50`
  - full workload: 100 no-cache baselines plus 1000 SeaCache candidates
  - remote first-50 handoff: `handoff/openvid_first50_2xa800/`, split across two A800 GPUs as prompts `0-24` and `25-49`
- Existing dirty working tree at session start included modified handoff files, modified `PROGRESS.md`, untracked SeaCache OpenVid experiment scripts, and a prompt-02 high-threshold result symlink/log. This session only moved reports, read context, and added this progress/log record.
