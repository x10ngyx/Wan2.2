# 2026-06-08 ZEUS-threshold alignment check

- Reviewed experiments/zeus_threshold_50step_45f_480p/run_experiments.sh against the completed ZEUS timestep experiment settings and command records.
- Confirmed shared settings are aligned: t2v-A14B, 832*480, 45 frames, 50 steps, dpm++, fixed seed 42, the same 10 prompts from prompt.txt, offload_model=True, convert_model_dtype, and disabled block/CFG cache.
- Confirmed threshold experiments use the same compute-only timing log field and the same FFmpeg PSNR helper as the formal ZEUS run.
- Confirmed threshold summary uses unique timestep counts from skipping_path/recompute_path, with separate branch-call count columns, matching the corrected ZEUS summary convention.
- Confirmed the intentional method difference: zeus-threshold does not use fixed ZEUS denominator/modular/lagrange cadence; it uses latent relative-L1 threshold decisions inside the same (model_stage, branch) state structure.
- Validation commands passed: bash -n for threshold run_experiments.sh, conda py_compile for threshold summarizer, PSNR helper, cache implementation, generate.py, and wan/text2video.py; prompt parser returned 10 prompts.
