# 2026-06-08 timestep-aware interp implementation

- Added a threshold-only caching mode named timestep_aware_interp.
- Existing ZEUS and existing threshold modes reuse_interp, interp_all, and reuse_all keep their original behavior.
- The new mode records recent recompute step indices in ZeusThresholdTimestepCacheState. On a skipped step it computes scale = (current_step - last_recompute_step) / (last_recompute_step - previous_recompute_step), then returns last + (last - previous) * scale.
- Exposed the new mode through generate.py and experiments/zeus_threshold_50step_45f_480p/run_batch.py. The shell script can pass it via ZEUS_CACHING_MODE=timestep_aware_interp.
- Validation passed: conda py_compile for timestep cache, generate.py, batch runner, and summarizer; generate.py --help and run_batch.py --help include the new mode.
- CPU dummy test passed: with true recompute outputs step1=10 and step4=40, timestep_aware_interp at step5 returns 50, while unchanged reuse_interp returns 70.
