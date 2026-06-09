# 2026-06-09 Block-cache stage clear summary fix

- Rechecked OOM risk after user asked whether block cache stores only the previous step. It does not store all historical steps, but BWCache keeps full previous per-block features for each active `(stage, branch)`, so memory remains large.
- Moved block-cache stage clearing before `_prepare_model_for_timestep()` so dead high-stage cache is freed before low model transfer.
- Updated BWCache and block-group cache `clear_stage()` to archive summaries before deleting tensor-heavy state, preserving reuse/recompute traces for final result tables.
- Validation: conda py_compile passed for touched files; CPU check verified cleared stages are removed from active tensor state while archived summaries remain visible in `summary()`.
