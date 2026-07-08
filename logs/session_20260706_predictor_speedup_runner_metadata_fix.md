# Session 2026-07-06 Predictor Speedup Runner Metadata Fix

Context: user accepted old checkpoint incompatibility as expected because new predictors will be trained, and asked to fix the implementation issues from the review.

Changes made:
- Fixed `adaptive_threshold_predictor/data.py` by importing `Iterable` from `collections.abc` alongside `Sequence`.
- Added explicit `--target_speedup` CLI argument, default `2.0`, to adaptive SeaCache historical runners:
  - `experiments/adaptive_seacache_ali_prompt12_50step_45f_480p/run_batch.py`
  - `experiments/adaptive_seacache_train10_50step_45f_480p/run_batch.py`
  - `experiments/adaptive_seacache_train15_test5_50step_45f_480p/run_batch.py`
  - `experiments/adaptive_seacache_overhead_train5_50step_45f_480p/run_batch.py`
  - `experiments/adaptive_seacache_mini_dit_split_compare_50step_45f_480p/run_batch.py`
- Replaced hard-coded online `target_speedup=2.0` with `args.target_speedup`.
- Archived target speedup in target env files, experiment config, command records, failed records, summary rows, and MiniDiT aggregate rows.
- Updated `PROGRESS.md` known issue text to reflect that runner metadata is now fixed with a single-value `--target_speedup` option.

Validation:
- `python -m compileall adaptive_threshold_predictor/data.py` plus the five modified runner scripts.
- `ast.parse` on all modified Python files.
- `git diff --check`.
- `--help` smoke checks confirmed `--target_speedup` appears for representative runners.

Notes:
- No GPU inference was run in this session.
- Old two-condition checkpoints remain incompatible by design; retrain speedup-conditioned predictors before launching new online runs.
