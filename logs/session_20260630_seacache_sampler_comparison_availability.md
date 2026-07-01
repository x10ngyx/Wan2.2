# 2026-06-30 SeaCache Sampler Comparison Availability

Task: answer whether complete SeaCache `dpm++` vs `unipc` performance comparison results exist.

Checked:
- `PROGRESS.md`
- `/hy-tmp/wan22_seacache_unipc_ali10_50step_45f_480p_20260627_023222/results/`
- `/hy-tmp/wan22_seacache_unipc_vbench10_50step_45f_480p_20260627_023222/results/`
- `/hy-tmp/wan22_vbench10_three_cache_full_merge_and_timestep_only_full_20260623/timestep_only_seacache_vbench10_full/wan22_seacache_vbench10_50step_45f_480p_20260618_161845/merged/`
- older Ali prompt-01/02 SeaCache DPM++ roots.

Conclusion:
- Full VBench10 SeaCache sampler comparison exists for overlapping thresholds `0.10`, `0.20`, `0.30`, and `0.50`.
- Full Ali-10 comparison does not exist. Ali-10 has full `unipc`, but `dpm++` has only prompt-01/02 pilot results.

No code or experiment data was changed. `PROGRESS.md` was updated with this lookup result.
