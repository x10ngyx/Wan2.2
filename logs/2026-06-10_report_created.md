# 2026-06-10 Report Creation

- Scanned `/hy-tmp/work/Wan2.2/experiment_results` and整理了有信息量的实验：completed formal runs、threshold sweeps、block-cache-only、three-cache development runs、cache ablation、以及 64 组合 three-cache threshold grid。
- Created `/hy-tmp/work/Wan2.2/report.md` with experiment configs, result tables, completeness notes, failed/superseded run notes, conclusions, and recommended next experiments.
- Key report conclusions: fixed ZEUS cadence is still the strongest validated 10-prompt result (`1.986x / 23.705 dB`); current prompt 01 three-cache follow-up candidates are `ts=0.005, block=0.015, cfg=0.03` and `ts=0.005, block=0.03, cfg=0.02`.
- Current planned three-cache experiments should keep CFG miss forced recompute disabled and use a single-process batch runner to avoid repeated checkpoint loading.

## Split Reports

- Added `/hy-tmp/work/Wan2.2/report_main_experiments.md` for the three primary experiments requested by the user: fixed ZEUS 10 prompts, ZEUS-threshold reuse_interp 10 prompts, and the three-cache 64-threshold prompt 01 grid.
- The main report fully expands all result rows with PSNR, compute time, and speedup: 10 fixed-ZEUS rows, 50 ZEUS-threshold rows, and 64 grid rows.
- Added `/hy-tmp/work/Wan2.2/report_supplementary_experiments.md` for the remaining informative experiments: smoke validation, prompt 01 threshold pilots, timestep-aware interpolation, block-cache-only, three-cache development reruns, cache ablation, and failed/superseded notes.
- Kept the original combined `/hy-tmp/work/Wan2.2/report.md` unchanged.
