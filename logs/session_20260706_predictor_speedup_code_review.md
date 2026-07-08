# Session 2026-07-06 Predictor Speedup Code Review

Scope: reviewed the predictor architecture/training pipeline changes that add `target_speedup` as a condition input and remove `target_oracle` mode.

Checks run:
- `git status --short`
- `git diff --stat`
- `rg -n "target_oracle|DATASET_MODES|dataset_mode|--dataset_mode" adaptive_threshold_predictor adaptive_seacache_wan22 experiments reports README.md PROGRESS.md`
- `python -m compileall adaptive_threshold_predictor adaptive_seacache_wan22`
- checkpoint shape probe for old MiniDiT checkpoint vs new speedup-conditioned smoke checkpoint.

Findings:
- Functional predictor/adaptive code no longer exposes `target_oracle`, `DATASET_MODES`, or `--dataset_mode`; remaining hits are historical report text and unrelated aggregate filenames.
- `target_speedup` is threaded through raw, packed, cached-feature, grid-feature datasets, model forward calls, metrics, and online adaptive gate config.
- Issue: `adaptive_threshold_predictor/data.py` still uses `Iterable` in annotations without importing it. Compile passes because annotations are postponed, but `typing.get_type_hints`/static tooling will fail.
- Issue: historical experiment runners now pass `target_speedup=2.0` but still default to old two-condition checkpoints. Those checkpoints have `cond_embed.0.weight` shape like `(96, 2)`, while the current models instantiate `(dim, 3)` and will not load. Existing two-condition checkpoints must be rejected or replaced with retrained speedup-conditioned checkpoints.
- Issue: historical adaptive runner summary/command metadata generally does not record the configured `target_speedup`, making new runs ambiguous if multiple speedup targets are compared.

No code changes were made in this review session.
