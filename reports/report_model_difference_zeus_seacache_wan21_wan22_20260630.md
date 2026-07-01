# Wan2.1 vs Wan2.2 Model Difference Results

Date: 2026-06-30

# ZEUS

## ZEUS Experiment Configuration

| Experiment | Code / result path | Model | Prompt set | Solver | Seed | Size | Frames | Steps | Shift | Guidance | Cache / schedule |
| --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| Wan2.1 official demo | `/hy-tmp/work/zeus-official-new/outputs/vbench10_wan21_zeus_seed42` | `/hy-tmp/models/Wan2.1-T2V-14B-Diffusers` | VBench10 | FlowMatch Euler | 42 | `832*480` | 45 | 50 | default | `5.0` | official ZEUS patch |
| Wan2.1 strict Euler | `/hy-tmp/work/Wan2.1-official-zeus-strict/outputs/vbench10_strict_zeus_euler_seed42` | `/hy-tmp/models/Wan2.1-T2V-14B` | VBench10 | Euler | 42 | `832*480` | 45 | 50 | `1.0` | `5.0` | strict guided ZEUS, `27/23` reuse/recompute |
| Wan2.1 strict UniPC | `/hy-tmp/work/Wan2.1-official-zeus-strict/outputs/vbench10_strict_zeus_seed42` | `/hy-tmp/models/Wan2.1-T2V-14B` | VBench10 | `unipc` | 42 | `832*480` | 45 | 50 | `1.0` | `5.0` | strict guided ZEUS, `27/23` reuse/recompute |
| Wan2.2 strict UniPC high/low reset | `/hy-tmp/wan22_strict_zeus_vbench10_unipc_50step_45f_480p_highlow_reset_20260629_1648` | `/hy-tmp/models/Wan2.2-T2V-A14B` | VBench10 | `unipc` | 42 | `832*480` | 45 | 50 | `12.0` | `(3.0, 4.0)` | strict guided ZEUS, `26/24` reuse/recompute |

## ZEUS Schedule Configuration

| Field | Value |
| --- | --- |
| `acc_range` | `(8, 47)` |
| `denominator` | `3` |
| `modular` | `(0, 1)` |
| `max_interval` | `6` |
| `lagrange_term` | `4` |
| `lagrange_int` | `4` |
| `lagrange_step` | `24` |

## ZEUS Aggregate Results

| Experiment | Model / stack | Solver | Pairs | Baseline total (s) | ZEUS total (s) | Speedup | Mean PSNR | Min mean PSNR | Max mean PSNR | Global min PSNR | Global max PSNR |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Wan2.1 official demo | Wan2.1 Diffusers + official patch | FlowMatch Euler | 10 | 4854.470 | 2258.600 | 2.1493x | 30.0372 | 20.1660 | 37.4820 | 18.53 | 42.17 |
| Wan2.1 strict Euler | Wan2.1 official code | Euler | 10 | 5376.029 | 2860.008 | 1.8797x | 32.8156 | 22.6882 | 39.0847 | - | - |
| Wan2.1 strict UniPC | Wan2.1 official code | `unipc` | 10 | 5552.712 | 2810.613 | 1.9756x | 30.4504 | 21.6556 | 38.1949 | - | - |
| Wan2.2 strict UniPC high/low reset | Wan2.2 official code | `unipc` | 10 | 5435.022 | 2619.430 | 2.0749x | 23.6675 | 18.7207 | 31.2300 | 17.00 | 35.01 |

## ZEUS Wan2.1 Official Demo Per-Sample Results

| Sample | Speedup | Mean PSNR | Min PSNR | Max PSNR |
| --- | ---: | ---: | ---: | ---: |
| `vbench10_001` | 2.1405x | 31.3802 | 28.92 | 33.93 |
| `vbench10_002` | 2.1415x | 34.5489 | 31.54 | 36.65 |
| `vbench10_003` | 2.1547x | 23.4878 | 22.30 | 25.62 |
| `vbench10_004` | 2.1534x | 29.7071 | 27.32 | 32.26 |
| `vbench10_005` | 2.1533x | 20.1660 | 18.53 | 22.29 |
| `vbench10_006` | 2.1505x | 32.8369 | 29.38 | 36.14 |
| `vbench10_007` | 2.1514x | 37.4820 | 34.98 | 42.17 |
| `vbench10_008` | 2.1495x | 27.6820 | 25.53 | 28.90 |
| `vbench10_009` | 2.1496x | 33.1513 | 31.78 | 34.21 |
| `vbench10_010` | 2.1490x | 29.9293 | 29.12 | 30.32 |

## ZEUS Wan2.1 Strict Euler Per-Sample Results

| Sample | Speedup | Mean PSNR | Min PSNR | Max PSNR |
| --- | ---: | ---: | ---: | ---: |
| `vbench10_001` | 1.9683x | 31.8520 | 29.51 | 34.00 |
| `vbench10_002` | 1.8873x | 37.7731 | 30.90 | 41.52 |
| `vbench10_003` | 1.9210x | 22.9907 | 21.37 | 24.53 |
| `vbench10_004` | 1.7989x | 32.0436 | 28.88 | 33.94 |
| `vbench10_005` | 1.8985x | 22.6882 | 20.12 | 26.00 |
| `vbench10_006` | 1.7869x | 38.4849 | 34.71 | 42.26 |
| `vbench10_007` | 1.9408x | 39.0847 | 33.71 | 45.54 |
| `vbench10_008` | 1.7898x | 34.7284 | 30.85 | 37.09 |
| `vbench10_009` | 1.9311x | 33.7909 | 31.56 | 35.81 |
| `vbench10_010` | 1.8991x | 34.7198 | 33.06 | 35.47 |

## ZEUS Wan2.1 Strict UniPC Per-Sample Results

| Sample | Speedup | Mean PSNR | Min PSNR | Max PSNR |
| --- | ---: | ---: | ---: | ---: |
| `vbench10_001` | 2.1004x | 30.4827 | 26.29 | 34.75 |
| `vbench10_002` | 2.1616x | 37.2342 | 31.21 | 40.99 |
| `vbench10_003` | 1.9697x | 23.0224 | 20.88 | 26.01 |
| `vbench10_004` | 1.9401x | 29.4993 | 27.70 | 31.33 |
| `vbench10_005` | 1.9915x | 21.6556 | 18.84 | 26.07 |
| `vbench10_006` | 1.9463x | 34.0627 | 29.35 | 38.12 |
| `vbench10_007` | 1.8966x | 38.1949 | 35.08 | 42.65 |
| `vbench10_008` | 1.9653x | 29.2051 | 26.99 | 30.69 |
| `vbench10_009` | 1.8259x | 31.9711 | 29.91 | 33.38 |
| `vbench10_010` | 1.9906x | 29.1762 | 28.53 | 30.63 |

## ZEUS Wan2.2 Strict UniPC High/Low Reset Per-Sample Results

| Sample | Baseline time (s) | ZEUS time (s) | Speedup | Mean PSNR | Min PSNR | Max PSNR | Reuse/Recompute |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `vbench10_001` | 543.531 | 262.109 | 2.0737x | 20.1033 | 17.00 | 22.80 | 26/24 |
| `vbench10_002` | 544.224 | 261.907 | 2.0779x | 26.2687 | 19.76 | 30.49 | 26/24 |
| `vbench10_003` | 544.378 | 262.517 | 2.0737x | 18.9731 | 17.30 | 20.92 | 26/24 |
| `vbench10_004` | 544.915 | 262.336 | 2.0772x | 23.6547 | 22.83 | 24.08 | 26/24 |
| `vbench10_005` | 544.224 | 262.612 | 2.0724x | 18.7207 | 17.33 | 19.41 | 26/24 |
| `vbench10_006` | 544.593 | 262.491 | 2.0747x | 21.6251 | 18.93 | 25.56 | 26/24 |
| `vbench10_007` | 544.334 | 262.400 | 2.0744x | 31.2300 | 28.07 | 35.01 | 26/24 |
| `vbench10_008` | 540.107 | 261.146 | 2.0682x | 28.4349 | 26.21 | 29.57 | 26/24 |
| `vbench10_009` | 542.789 | 261.026 | 2.0794x | 21.0478 | 20.20 | 21.86 | 26/24 |
| `vbench10_010` | 541.927 | 260.886 | 2.0773x | 26.6169 | 25.86 | 27.20 | 26/24 |

## ZEUS Wan2.1 Strict UniPC vs Wan2.2 Strict UniPC Aggregate Comparison

| Metric | Wan2.1 strict UniPC | Wan2.2 strict UniPC | Wan2.2 - Wan2.1 |
| --- | ---: | ---: | ---: |
| Baseline total (s) | 5552.712 | 5435.022 | -117.690 |
| ZEUS total (s) | 2810.613 | 2619.430 | -191.183 |
| Speedup | 1.9756x | 2.0749x | +0.0993x |
| Mean PSNR | 30.4504 | 23.6675 | -6.7829 |
| Min mean PSNR | 21.6556 | 18.7207 | -2.9349 |
| Max mean PSNR | 38.1949 | 31.2300 | -6.9649 |
| Reuse/Recompute | 27/23 | 26/24 | - |

## ZEUS Wan2.1 Strict UniPC vs Wan2.2 Strict UniPC Per-Sample Comparison

| Sample | Wan2.1 speedup | Wan2.2 speedup | Wan2.1 PSNR | Wan2.2 PSNR | PSNR delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `vbench10_001` | 2.1004x | 2.0737x | 30.4827 | 20.1033 | -10.3793 |
| `vbench10_002` | 2.1616x | 2.0779x | 37.2342 | 26.2687 | -10.9656 |
| `vbench10_003` | 1.9697x | 2.0737x | 23.0224 | 18.9731 | -4.0493 |
| `vbench10_004` | 1.9401x | 2.0772x | 29.4993 | 23.6547 | -5.8447 |
| `vbench10_005` | 1.9915x | 2.0724x | 21.6556 | 18.7207 | -2.9349 |
| `vbench10_006` | 1.9463x | 2.0747x | 34.0627 | 21.6251 | -12.4376 |
| `vbench10_007` | 1.8966x | 2.0744x | 38.1949 | 31.2300 | -6.9649 |
| `vbench10_008` | 1.9653x | 2.0682x | 29.2051 | 28.4349 | -0.7702 |
| `vbench10_009` | 1.8259x | 2.0794x | 31.9711 | 21.0478 | -10.9233 |
| `vbench10_010` | 1.9906x | 2.0773x | 29.1762 | 26.6169 | -2.5593 |

# SeaCache

## SeaCache Experiment Configuration

| Experiment | Code / result path | Model | Prompt set | Solver | Seed | Size | Frames | Steps | Shift | Guidance | Cache / threshold |
| --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | --- | --- | --- |
| Wan2.1 SeaCache Ali-10 | `/hy-tmp/wan21_seacache_threshold_compare_ali10_seed42_832x480_45f_50steps` | `/hy-tmp/models/Wan2.1-T2V-14B` | Ali-10 | `unipc` | 42 | `832*480` | 45 | 50 | default | default | reference `0.0`, candidate `0.20` |
| Wan2.2 SeaCache Ali-10 | `/hy-tmp/wan22_seacache_ali10_unipc_50step_45f_480p_20260629_224430` | `/hy-tmp/models/Wan2.2-T2V-A14B` | Ali-10 | `unipc` | 42 | `832*480` | 45 | 50 | `12.0` | `(3.0, 4.0)` | no-cache baseline, candidate `0.20` |

## SeaCache Wan2.2 Cache Configuration

| Field | Value |
| --- | --- |
| `timestep_cache` | `seacache` |
| `seacache_threshold` | `0.20` |
| `seacache_use_ret_steps` | `false` |
| `seacache_power_exp` | `3.0` |
| `seacache_power_const` | `1.0` |
| `seacache_eps` | `1e-16` |
| `seacache_norm_mode` | `mean` |
| `block_cache` | `none` |
| `cfg_cache` | `none` |
| `timing_source` | `inference_compute_elapsed_seconds` |

## SeaCache Aggregate Results

| Experiment | Reference definition | Candidate definition | Pairs | Reference total (s) | Candidate total (s) | Speedup | Mean PSNR | Min mean PSNR | Max mean PSNR | Global min PSNR | Reuse/Recompute |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Wan2.1 SeaCache Ali-10 | `seacache_thresh=0.0` | `seacache_thresh=0.20` | 10 | 5410.653 | 2910.806 | 1.8588x | 30.1260 | 23.4300 | 40.7540 | - | - |
| Wan2.2 SeaCache Ali-10 | no-cache baseline | timestep-only SeaCache `threshold=0.20` | 10 | 5421.520 | 3440.106 | 1.5760x | 27.5130 | 22.4287 | 39.6138 | 21.04 | 400/600 |

## SeaCache Wan2.1 Ali-10 Per-Sample Results

| Sample | Reference time (s) | Candidate time (s) | Speedup | Mean PSNR | Min PSNR | Max PSNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ali_001` | 537.240 | 287.419 | 1.8692x | 27.8998 | 25.02 | 29.10 |
| `ali_002` | 540.441 | 290.773 | 1.8586x | 40.7540 | 38.14 | 41.95 |
| `ali_003` | 541.403 | 293.166 | 1.8467x | 32.6547 | 30.27 | 35.82 |
| `ali_004` | 539.316 | 290.860 | 1.8542x | 26.7629 | 25.67 | 27.91 |
| `ali_005` | 538.806 | 290.011 | 1.8579x | 40.3633 | 39.84 | 41.20 |
| `ali_006` | 540.864 | 290.709 | 1.8605x | 23.4300 | 21.75 | 25.33 |
| `ali_007` | 542.304 | 292.767 | 1.8523x | 32.7887 | 30.47 | 35.53 |
| `ali_008` | 547.581 | 294.673 | 1.8583x | 25.5924 | 22.65 | 28.94 |
| `ali_009` | 542.596 | 290.518 | 1.8677x | 27.1311 | 25.66 | 27.90 |
| `ali_010` | 540.101 | 289.910 | 1.8630x | 23.8831 | 21.54 | 27.23 |

## SeaCache Wan2.2 Ali-10 Per-Sample Results

| Sample | Baseline time (s) | SeaCache time (s) | Speedup | Mean PSNR | Min PSNR | Max PSNR | Reuse/Recompute |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ali_001` | 541.223 | 344.302 | 1.5719x | 23.9542 | 22.51 | 26.44 | 40/60 |
| `ali_002` | 542.542 | 344.006 | 1.5771x | 29.7913 | 25.95 | 30.93 | 40/60 |
| `ali_003` | 542.246 | 343.966 | 1.5765x | 31.5704 | 30.73 | 32.48 | 40/60 |
| `ali_004` | 542.319 | 343.834 | 1.5773x | 23.6969 | 21.04 | 24.48 | 40/60 |
| `ali_005` | 542.082 | 343.816 | 1.5767x | 39.6138 | 38.51 | 40.98 | 40/60 |
| `ali_006` | 541.883 | 344.160 | 1.5745x | 23.2389 | 21.74 | 24.10 | 40/60 |
| `ali_007` | 542.224 | 343.750 | 1.5774x | 30.3627 | 28.50 | 31.21 | 40/60 |
| `ali_008` | 542.204 | 344.101 | 1.5757x | 24.5182 | 21.53 | 26.18 | 40/60 |
| `ali_009` | 542.399 | 344.321 | 1.5753x | 22.4287 | 21.35 | 23.25 | 40/60 |
| `ali_010` | 542.398 | 343.850 | 1.5774x | 25.9553 | 24.10 | 27.84 | 40/60 |

## SeaCache Wan2.1 vs Wan2.2 Aggregate Comparison

| Metric | Wan2.1 SeaCache | Wan2.2 SeaCache | Wan2.2 - Wan2.1 |
| --- | ---: | ---: | ---: |
| Reference total (s) | 5410.653 | 5421.520 | +10.867 |
| Candidate total (s) | 2910.806 | 3440.106 | +529.300 |
| Speedup | 1.8588x | 1.5760x | -0.2828x |
| Mean PSNR | 30.1260 | 27.5130 | -2.6130 |
| Min mean PSNR | 23.4300 | 22.4287 | -1.0013 |
| Max mean PSNR | 40.7540 | 39.6138 | -1.1402 |

## SeaCache Wan2.1 vs Wan2.2 Per-Sample Comparison

| Sample | Wan2.1 speedup | Wan2.2 speedup | Speedup delta | Wan2.1 PSNR | Wan2.2 PSNR | PSNR delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ali_001` | 1.8692x | 1.5719x | -0.2972x | 27.8998 | 23.9542 | -3.9456 |
| `ali_002` | 1.8586x | 1.5771x | -0.2815x | 40.7540 | 29.7913 | -10.9627 |
| `ali_003` | 1.8467x | 1.5765x | -0.2703x | 32.6547 | 31.5704 | -1.0842 |
| `ali_004` | 1.8542x | 1.5773x | -0.2769x | 26.7629 | 23.6969 | -3.0660 |
| `ali_005` | 1.8579x | 1.5767x | -0.2812x | 40.3633 | 39.6138 | -0.7496 |
| `ali_006` | 1.8605x | 1.5745x | -0.2860x | 23.4300 | 23.2389 | -0.1911 |
| `ali_007` | 1.8523x | 1.5774x | -0.2750x | 32.7887 | 30.3627 | -2.4260 |
| `ali_008` | 1.8583x | 1.5757x | -0.2826x | 25.5924 | 24.5182 | -1.0742 |
| `ali_009` | 1.8677x | 1.5753x | -0.2924x | 27.1311 | 22.4287 | -4.7024 |
| `ali_010` | 1.8630x | 1.5774x | -0.2856x | 23.8831 | 25.9553 | +2.0722 |
