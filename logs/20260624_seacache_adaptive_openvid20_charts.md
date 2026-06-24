# 2026-06-24 SeaCache/Adaptive OpenVid20 Charts

## What Changed

- Added `reports/plot_seacache_adaptive_train15_test5.py`.
- Generated Pareto-style charts matching the existing
  `reports/assets/vbench10_three_cache/` visual pattern.

## Inputs

- Fixed SeaCache:
  `/hy-tmp/wan22_seacache_train15_test5_50step_45f_480p_20260623_160513/results/summary.csv`
- Adaptive SeaCache:
  `/hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/results/summary.csv`

## Outputs

Output directory:

```text
reports/assets/seacache_adaptive_train15_test5/
```

Chart sets, each written as PNG/PDF/SVG:

- `openvid20_fixed_seacache_pareto_scatter`
- `openvid20_adaptive_seacache_pareto_scatter`
- `openvid20_fixed_vs_adaptive_seacache_pareto_overlay`

Additional plot data:

- `openvid20_fixed_seacache_plot_aggregate.csv`
- `openvid20_adaptive_seacache_plot_aggregate.csv`
- `plot_inputs.json`

## Validation

- `python -m py_compile reports/plot_seacache_adaptive_train15_test5.py` passed.
- The plotting script ran successfully with the Wan2.2 conda Python.
- Visual inspection of the overlay PNG passed.
- Initial overlay generation exposed an overly long source-note bbox; the source
  note was shortened and charts were regenerated.

## Metric Note

Aggregate speedup in these plots is computed as:

```text
sum(baseline_compute_elapsed_seconds) / sum(candidate_compute_elapsed_seconds)
```

This matches the fixed SeaCache aggregate convention and can differ slightly
from report tables that average per-prompt speedup values.

## Train Split Addendum

Added train-split-only outputs to the same asset directory:

- `openvid20_train_fixed_seacache_results.csv`
  - 60 rows: 15 train prompts x 4 fixed thresholds.
- `openvid20_train_adaptive_seacache_results.csv`
  - 45 rows: 15 train prompts x 3 adaptive target PSNRs.
- `openvid20_train_fixed_seacache_aggregate.csv`
- `openvid20_train_adaptive_seacache_aggregate.csv`
- `openvid20_train_fixed_vs_adaptive_seacache_aggregate.csv`
- `openvid20_train_fixed_vs_adaptive_seacache_pareto_overlay.{png,pdf,svg}`

Train-split aggregate values:

| method | setting | speedup | mean PSNR |
| --- | --- | ---: | ---: |
| fixed SeaCache | threshold 0.1 | 1.137x | 42.483 dB |
| fixed SeaCache | threshold 0.2 | 1.607x | 30.821 dB |
| fixed SeaCache | threshold 0.4 | 2.468x | 23.794 dB |
| fixed SeaCache | threshold 0.6 | 3.177x | 20.634 dB |
| adaptive SeaCache | target 20 | 3.159x | 20.481 dB |
| adaptive SeaCache | target 25 | 2.387x | 23.137 dB |
| adaptive SeaCache | target 30 | 1.840x | 27.477 dB |

Validation:

- Plotting script ran successfully after the train-split extension.
- `python -m py_compile reports/plot_seacache_adaptive_train15_test5.py` passed.
- Visual inspection of the train overlay passed.
- Temporary `reports/__pycache__/` was removed.
