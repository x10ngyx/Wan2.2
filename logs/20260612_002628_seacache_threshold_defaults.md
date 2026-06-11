# 2026-06-12 SeaCache threshold defaults

- User requested 10 SeaCache thresholds covering the full `0.1` to `0.8` interval, with no higher-threshold expansion.
- Set the default threshold list to `0.10 0.15 0.20 0.25 0.30 0.40 0.50 0.60 0.70 0.80`.
- Synchronized defaults in the regular SeaCache runner, OpenVid-100 SeaCache runner, and first-50 two-A800 handoff launcher/docs.
- Validation passed:
  - conda `py_compile` for SeaCache/OpenVid runners and summarizers plus handoff merger.
  - `bash -n` for regular SeaCache, OpenVid SeaCache, and handoff launch/collection scripts.
  - regular SeaCache CPU validation: `prompt_count=1`, `expected_candidate_runs=10`.
  - OpenVid SeaCache CPU validation: `threshold_count=10`, `expected_candidate_runs=1000`.
