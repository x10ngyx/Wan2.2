# 2026-06-08 Reuse/Recompute Count Fix

- Investigated zeus_reuse_count and zeus_recompute_count after the formal ZEUS timestep run.
- Root cause: summarize_results.py summed every cache state, so high/low and cond/uncond branch states were reported as one combined count. For timestep-level reporting this doubled the apparent reuse/recompute counts.
- Updated experiments/zeus_timestep_cache_50step_45f_480p/summarize_results.py and experiments/zeus_threshold_50step_45f_480p/summarize_results.py.
- Existing reuse/recompute count columns now report unique timestep counts derived from skipping_path and recompute_path.
- Added branch-call count columns to preserve summed cond/uncond model-call counts.
- Validation: py_compile passed for both summarizers. Regenerated the completed formal ZEUS summary; prompt 01-03 show 25 reuse / 25 recompute timesteps and 50 / 50 branch calls. Totals across 10 prompts are 250 / 250 timesteps and 500 / 500 branch calls.
- Backed up the pre-fix completed-run CSV to /hy-tmp/wan22_zeus_timestep_cache_50step_45f_480p_full_20260608_114307/results/summary.branch_count_before_fix.csv.
