# 2026-06-19 SeaCache VBench10 Completion

## Summary

Completed the timestep-cache-only SeaCache VBench10 experiment using the same 10 thresholds as the OpenVid run.

Experiment root:

```text
/hy-tmp/wan22_seacache_vbench10_50step_45f_480p_20260618_161845
```

## Configuration

- Prompt set: `/hy-tmp/work/Wan2.2/test_sets/Vbench10/prompts.jsonl`
- Model: `/hy-tmp/models/Wan2.2-T2V-A14B`
- Python/runtime: `/hy-tmp/env/Wan2.2`
- ffmpeg/ffprobe path used: `/hy-tmp/env/Wan2.2/bin`
- Thresholds: `0.10 0.15 0.20 0.25 0.30 0.40 0.50 0.60 0.70 0.80`
- Split: GPU0 handled prompts 1-5, GPU1 handled prompts 6-10.

## Completion

- Baselines: `10/10`
- SeaCache candidates: `100/100`
- PSNR JSON files: `100/100`
- ffprobe JSON files: `110/110`
- Failed records: `0`
- GPU state after completion: both GPUs idle.

Merged outputs:

```text
/hy-tmp/wan22_seacache_vbench10_50step_45f_480p_20260618_161845/merged/summary.csv
/hy-tmp/wan22_seacache_vbench10_50step_45f_480p_20260618_161845/merged/aggregate_by_threshold.csv
/hy-tmp/wan22_seacache_vbench10_50step_45f_480p_20260618_161845/merged/aggregate_by_threshold.json
```

## Aggregate Results

| Threshold | Speedup | Mean PSNR | Min PSNR |
| --- | ---: | ---: | ---: |
| 0.10 | 1.108x | 35.897 dB | 27.97 dB |
| 0.15 | 1.410x | 28.001 dB | 18.52 dB |
| 0.20 | 1.574x | 25.958 dB | 17.53 dB |
| 0.25 | 1.837x | 24.969 dB | 18.23 dB |
| 0.30 | 1.979x | 23.515 dB | 16.94 dB |
| 0.40 | 2.424x | 20.601 dB | 14.40 dB |
| 0.50 | 2.744x | 20.679 dB | 13.69 dB |
| 0.60 | 3.125x | 20.659 dB | 14.16 dB |
| 0.70 | 3.339x | 19.838 dB | 14.46 dB |
| 0.80 | 3.535x | 19.657 dB | 14.20 dB |
