# Session 2026-07-08 Adaptive Threshold Predictor Handoff Review

## Summary

- Reviewed `adaptive_threshold_predictor/` for handoff readability.
- Added a file map and workflow notes to `adaptive_threshold_predictor/README.md`.
- Clarified that current offline training uses `candidate_inverse` labels where code fields named `target_psnr` and `target_speedup` are measured candidate outcomes.
- Added packed raw-latent cache, Grid MLP baseline, split-mode, and caveat documentation.
- Fixed misleading script text:
  - `inspect_trace_data.py` no longer describes threshold output as `[0,1]`.
  - `build_grid_feature_cache.py` now says it builds features for `grid_mlp`, not MiniDiT.
  - Removed unused `--resume_existing` from `build_raw_latent_cache.py`.

## Validation

- Ran static compile:
  `/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m compileall -q adaptive_threshold_predictor`
- Checked help output for:
  - `adaptive_threshold_predictor.train_gate`
  - `adaptive_threshold_predictor.build_raw_latent_cache`
  - `adaptive_threshold_predictor.build_grid_feature_cache`
- Ran trace smoke check:
  `/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m adaptive_threshold_predictor.inspect_trace_data --step_index 0 --target_psnr 30 --target_speedup 2.0`
- Ran a tiny CPU training smoke:
  `/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m adaptive_threshold_predictor.train_gate --epochs 1 --batch_size 2 --max_examples 4 --device cpu --num_workers 0 --out_dir /hy-tmp/wan22_adaptive_threshold_predictor_doccheck`
- Removed the temporary smoke output directory after validation.

## Notes

- No GPU training or inference was run.
- The package is clearer for handoff, but full experiment claims should still point to `reports/report_predictor_speedup.md` and archived result roots.
