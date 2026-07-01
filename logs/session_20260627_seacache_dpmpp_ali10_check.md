# 2026-06-27 SeaCache DPM++ Ali-10 Check

Task: answer whether full `SeaCache + dpm++ + ali_10` performance had already been measured.

Checked:
- `PROGRESS.md`
- `reports/`
- `logs/`
- `experiments/`
- `/hy-tmp/wan22_*` result roots
- `experiment_results/` symlinks

Conclusion:
- No complete formal full Ali-10 result root was found for `SeaCache + dpm++`.
- Existing `SeaCache + dpm++` Ali data covers only prompt 1 and prompt 2:
  - `/hy-tmp/wan22_seacache_50step_45f_480p_20260611_191733`
  - `/hy-tmp/wan22_seacache_prompt02_dense_20260611_204826`
  - `/hy-tmp/wan22_seacache_prompt02_highthr_20260612_000218`
- Existing full Ali-10 SeaCache data is the new `unipc` run:
  - `/hy-tmp/wan22_seacache_unipc_ali10_50step_45f_480p_20260627_023222`

No code or experiment outputs were changed. `PROGRESS.md` was updated with this lookup result.
