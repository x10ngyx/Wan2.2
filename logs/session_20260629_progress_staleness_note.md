# 2026-06-29 Progress Staleness Note

- Read `PROGRESS.md` at session start as required.
- User pointed out that the assistant's "next recommended work" statement was stale because a first adaptive threshold predictor already exists.
- Verified repository evidence:
  - `adaptive_threshold_predictor/train_gate.py`
  - `adaptive_seacache_wan22/`
  - `reports/report_adaptive_predictor.md`
  - `reports/report_adaptive_seacache_train15_test5_and_overhead.md`
  - adaptive predictor/session logs from 2026-06-15 through 2026-06-24
- Updated `PROGRESS.md` with a note that the old "first adaptive-threshold predictor baseline" recommendation is stale.
- Current adaptive follow-up should follow `todo.md`: retest the existing predictor on VBench10 and diagnose why performance is poor.

Validation:
- No tests or experiments were run; this was a documentation/status correction only.
