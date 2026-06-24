# 2026-06-23 Project Compute Cost Scan

## Scope

- Scanned experiment result directories under `/hy-tmp`.
- Included canonical completed result summaries from `/hy-tmp/wan22_*`, `/hy-tmp/openvid_100_seacache_trace_data/data/tables/summary.csv`, and merged VBench summaries.
- Excluded aggregate tables, handoff build copies, repo mirror copies, interim duplicates, partial duplicates, VBench shard/per-prompt duplicates when merged summaries exist.

## Cost Rule

- User requested not to use raw inference time directly.
- Actual GPU occupancy estimate: `inference_compute_elapsed_seconds * 1.5`.
- A100 rate: `6 CNY/hour`.

## Result

- Parseable inference compute total: `222.267 GPU-hours`.
- Adjusted GPU occupancy: `333.401 A100-hours`.
- Estimated total cost: `2000.40 CNY`.
- Parsed run records after deduplication and limited incomplete-run supplementation: `2641`.

## Notes

- Baseline elapsed values repeated in candidate tables were counted once per sample within each result table.
- Empty/incomplete runs were supplemented from available `partial_summary.csv`, `summary_interim_prompt01.csv`, or logs with `inference_compute_elapsed_seconds`.
- Failed runs without any elapsed-time record, such as the OOM branch in the AdaCache smoke test, were not independently estimated beyond completed logged inference records.
