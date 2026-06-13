# 2026-06-13 OpenVid SeaCache Inspection

## Scope

Checked `/hy-tmp/wan22_seacache_openvid100_50step_45f_480p_20260612_002814` to determine whether the OpenVid prompts 76-100 SeaCache run completed after the previous handoff.

## Findings

- `runner.log` ends with `Completed experiment: /hy-tmp/wan22_seacache_openvid100_50step_45f_480p_20260612_002814`.
- `failed/` is empty.
- Baseline outputs: `25` MP4 files.
- SeaCache outputs: `25` MP4 files for each of `10` thresholds, `250` candidate MP4 files total.
- `ffprobe/`: `275` files.
- `psnr/`: `750` files.
- `logs/`: `551` files.
- `commands/`: `275` files.
- `results/summary.csv`: `250` candidate rows, `25` unique samples, `10` threshold labels.
- `results/aggregate_by_threshold.csv`: `10` aggregate threshold rows.

## Aggregate Results

| threshold | speedup | mean PSNR | min PSNR |
|---:|---:|---:|---:|
| 0.10 | 1.113x | 42.333 dB | 34.26 dB |
| 0.15 | 1.412x | 34.222 dB | 23.62 dB |
| 0.20 | 1.575x | 30.188 dB | 19.50 dB |
| 0.25 | 1.844x | 26.787 dB | 19.36 dB |
| 0.30 | 1.976x | 25.170 dB | 17.67 dB |
| 0.40 | 2.418x | 22.836 dB | 14.50 dB |
| 0.50 | 2.746x | 21.429 dB | 14.02 dB |
| 0.60 | 3.112x | 19.567 dB | 13.01 dB |
| 0.70 | 3.337x | 19.282 dB | 13.05 dB |
| 0.80 | 3.517x | 19.004 dB | 13.24 dB |

## Files Changed

- Updated `PROGRESS.md` with the verified completion state and aggregate metrics.
- Added this session log.

## Validation

Used shell counts over artifact directories plus CSV row counts via Python `csv.DictReader`. No GPU jobs were launched.

## Next

Use `results/summary.csv` and `results/aggregate_by_threshold.csv` as completed inputs for the consolidated threshold dataset.
