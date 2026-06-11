# 2026-06-11 SeaCache vs ZEUS-threshold prompt 01-02 comparison

- User requested a comparison of ZEUS-threshold and SeaCache performance on prompt 01 and prompt 02.
- Read ZEUS-threshold rows from `/hy-tmp/wan22_zeus_threshold_reuse_interp_10prompt_5th_20260608_195427/results/summary.csv`.
- Read SeaCache prompt 01 rows from `/hy-tmp/wan22_seacache_50step_45f_480p_20260611_191733/results/summary.csv`.
- Read SeaCache prompt 02 rows from `/hy-tmp/wan22_seacache_prompt02_dense_20260611_204826/results/summary.csv`.
- Wrote `report_seacache_vs_zeus_threshold_prompt12.md` with per-prompt tables and conclusions.
- Main conclusion: SeaCache has the better quality/speed frontier on both prompts. It is especially strong on prompt 02, where SeaCache `0.30` achieves `1.965x` at `29.582 dB`, while ZEUS-threshold around similar or faster speed remains around `20.4 dB`.
