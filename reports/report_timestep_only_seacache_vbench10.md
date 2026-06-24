# VBench10 Timestep-Only SeaCache Report

Generated: 2026-06-23 Asia/Shanghai.

## 1. Experiment Configuration

This experiment evaluates timestep-cache-only SeaCache on the 10-prompt VBench10 subset. There is no block cache and no CFG cache. Ten thresholds are swept, with two GPU shards each covering five prompts; the tables below use the merged 10-prompt result.

| setting | value |
| --- | --- |
| task | t2v-A14B |
| ckpt_dir | /hy-tmp/models/Wan2.2-T2V-A14B |
| prompt_path | /hy-tmp/work/Wan2.2/test_sets/Vbench10/prompts.jsonl |
| base_seed | 42 |
| size | 832*480 |
| frame_num | 45 |
| sample_steps | 50 |
| sample_solver | dpm++ |
| sample_shift | 12.0 |
| sample_guide_scale | (3.0, 4.0) |
| python_bin | /hy-tmp/env/Wan2.2/bin/python |
| ffprobe_bin | /hy-tmp/env/Wan2.2/bin/ffprobe |
| resume_existing | True |
| timestep_cache | seacache |
| block_cache | none |
| cfg_cache | none |
| seacache_num_steps | None |
| seacache_use_ret_steps | False |
| seacache_power_exp | 3.0 |
| seacache_power_const | 1.0 |
| seacache_eps | 1e-16 |
| seacache_norm_mode | mean |
| thresholds | 0.10 0.15 0.20 0.25 0.30 0.40 0.50 0.60 0.70 0.80 |
| expected_candidate_runs | 50 |
| shard_gpu0 | prompt_start=0, prompt_limit=5, selected_prompt_count=5, root=/hy-tmp/wan22_seacache_vbench10_50step_45f_480p_20260618_161845/shard_gpu0_p000_004 |
| shard_gpu1 | prompt_start=5, prompt_limit=5, selected_prompt_count=5, root=/hy-tmp/wan22_seacache_vbench10_50step_45f_480p_20260618_161845/shard_gpu1_p005_009 |


Source result files:

- full per-prompt table: `/hy-tmp/wan22_seacache_vbench10_50step_45f_480p_20260618_161845/merged/summary.csv`

- 10-prompt aggregate table: `/hy-tmp/wan22_seacache_vbench10_50step_45f_480p_20260618_161845/merged/aggregate_by_threshold.csv`


## Prompt Set

| sample_id | prompt |
| --- | --- |
| vbench10_001 | A woman is playing football. |
| vbench10_002 | A horse is running along the beach, then it suddenly stops and starts grazing. |
| vbench10_003 | The race began, and Team A quickly took the lead, securing the front position. Throughout the race, Team B kept pushing, gradually closing the gap on Team A, especially on the downhill segment. In the next uphill section, the Team C runner displayed strong endurance and stamina, overtaking both Team A and Team B in a few short kilometers, taking the lead. As the race entered its final stage, Team B increased their speed, but Team C maintained a steady pace and continued to lead. At the finish line, Team C sprinted to victory with a clear advantage. |
| vbench10_004 | Snow White was driven into the forest by the evil queen who was jealous of her beauty. In the forest, Snow White met seven kind dwarfs who offered her shelter. However, the evil queen disguised herself as an old woman and gave Snow White a poisoned apple. After biting the apple, Snow White fell into a deep sleep. At that moment, a prince heard of her plight and came to save her. He kissed her, breaking the spell. After Snow White regained consciousness, she decided to take revenge on the queen. She, along with the prince and the dwarfs, returned to the palace and overthrew the evil queen, restoring peace to the kingdom. |
| vbench10_005 | The camera orbits around in a clockwise direction. Forbidden City. |
| vbench10_006 | Equal amounts of yellow and blue paint are rapidly combined, with the mixture being vigorously stirred until fully blended. |
| vbench10_007 | A timelapse captures the reaction as concentrated sulfuric acid is poured onto a rubber balloon. |
| vbench10_008 | A cat is on the right of a rock, then the cat runs to the left of the rock. |
| vbench10_009 | people are playing ping-pong. |
| vbench10_010 | The camera orbits around. Serengeti, the camera circles around. |


## 2. Full Result Table

One row per `(prompt, threshold)`. Timing columns are seconds; `speedup = baseline_elapsed / seacache_elapsed`; larger `mean_psnr` is closer to the no-cache baseline.

| sample_id | threshold | label | base_s | seacache_s | speedup | mean_psnr | min_psnr | max_psnr | psnr_frames | reuse | recomp | reuse_branch | recomp_branch | shard |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vbench10_001 | 0.10 | th_0p10 | 538.211 | 482.917 | 1.115 | 32.608 | 28.460 | 36.510 | 45 | 6 | 44 | 12 | 88 | shard_gpu0_p000_004 |
| vbench10_002 | 0.10 | th_0p10 | 539.263 | 482.986 | 1.117 | 42.542 | 40.060 | 43.460 | 45 | 6 | 44 | 12 | 88 | shard_gpu0_p000_004 |
| vbench10_003 | 0.10 | th_0p10 | 539.507 | 492.713 | 1.095 | 31.720 | 27.970 | 33.960 | 45 | 5 | 45 | 10 | 90 | shard_gpu0_p000_004 |
| vbench10_004 | 0.10 | th_0p10 | 539.041 | 482.355 | 1.118 | 36.091 | 33.440 | 37.350 | 45 | 6 | 44 | 12 | 88 | shard_gpu0_p000_004 |
| vbench10_005 | 0.10 | th_0p10 | 538.963 | 492.044 | 1.095 | 30.030 | 28.400 | 30.990 | 45 | 5 | 45 | 10 | 90 | shard_gpu0_p000_004 |
| vbench10_006 | 0.10 | th_0p10 | 541.959 | 496.394 | 1.092 | 35.110 | 33.140 | 38.620 | 45 | 5 | 45 | 10 | 90 | shard_gpu1_p005_009 |
| vbench10_007 | 0.10 | th_0p10 | 543.609 | 486.295 | 1.118 | 46.819 | 42.980 | 49.490 | 45 | 6 | 44 | 12 | 88 | shard_gpu1_p005_009 |
| vbench10_008 | 0.10 | th_0p10 | 544.246 | 486.806 | 1.118 | 31.390 | 30.470 | 32.180 | 45 | 6 | 44 | 12 | 88 | shard_gpu1_p005_009 |
| vbench10_009 | 0.10 | th_0p10 | 544.039 | 496.682 | 1.095 | 32.880 | 31.630 | 33.980 | 45 | 5 | 45 | 10 | 90 | shard_gpu1_p005_009 |
| vbench10_010 | 0.10 | th_0p10 | 544.254 | 486.898 | 1.118 | 39.783 | 38.470 | 40.700 | 45 | 6 | 44 | 12 | 88 | shard_gpu1_p005_009 |
| vbench10_001 | 0.15 | th_0p15 | 538.211 | 380.381 | 1.415 | 23.148 | 18.520 | 26.210 | 45 | 16 | 34 | 32 | 68 | shard_gpu0_p000_004 |
| vbench10_002 | 0.15 | th_0p15 | 539.263 | 383.067 | 1.408 | 36.762 | 31.090 | 39.660 | 45 | 16 | 34 | 32 | 68 | shard_gpu0_p000_004 |
| vbench10_003 | 0.15 | th_0p15 | 539.507 | 383.258 | 1.408 | 23.275 | 20.310 | 26.700 | 45 | 16 | 34 | 32 | 68 | shard_gpu0_p000_004 |
| vbench10_004 | 0.15 | th_0p15 | 539.041 | 382.101 | 1.411 | 28.142 | 27.220 | 28.680 | 45 | 16 | 34 | 32 | 68 | shard_gpu0_p000_004 |
| vbench10_005 | 0.15 | th_0p15 | 538.963 | 382.256 | 1.410 | 22.948 | 21.950 | 23.380 | 45 | 16 | 34 | 32 | 68 | shard_gpu0_p000_004 |
| vbench10_006 | 0.15 | th_0p15 | 541.959 | 383.596 | 1.413 | 27.469 | 25.900 | 29.260 | 45 | 16 | 34 | 32 | 68 | shard_gpu1_p005_009 |
| vbench10_007 | 0.15 | th_0p15 | 543.609 | 385.742 | 1.409 | 37.683 | 34.260 | 45.120 | 45 | 16 | 34 | 32 | 68 | shard_gpu1_p005_009 |
| vbench10_008 | 0.15 | th_0p15 | 544.246 | 386.122 | 1.410 | 26.897 | 26.370 | 27.400 | 45 | 16 | 34 | 32 | 68 | shard_gpu1_p005_009 |
| vbench10_009 | 0.15 | th_0p15 | 544.039 | 385.454 | 1.411 | 23.737 | 21.920 | 25.160 | 45 | 16 | 34 | 32 | 68 | shard_gpu1_p005_009 |
| vbench10_010 | 0.15 | th_0p15 | 544.254 | 386.132 | 1.410 | 29.954 | 25.570 | 31.480 | 45 | 16 | 34 | 32 | 68 | shard_gpu1_p005_009 |
| vbench10_001 | 0.20 | th_0p20 | 538.211 | 341.811 | 1.575 | 20.295 | 17.530 | 23.430 | 45 | 20 | 30 | 40 | 60 | shard_gpu0_p000_004 |
| vbench10_002 | 0.20 | th_0p20 | 539.263 | 342.794 | 1.573 | 32.270 | 25.220 | 37.070 | 45 | 20 | 30 | 40 | 60 | shard_gpu0_p000_004 |
| vbench10_003 | 0.20 | th_0p20 | 539.507 | 342.285 | 1.576 | 20.250 | 17.550 | 23.470 | 45 | 20 | 30 | 40 | 60 | shard_gpu0_p000_004 |
| vbench10_004 | 0.20 | th_0p20 | 539.041 | 342.249 | 1.575 | 24.736 | 23.830 | 25.150 | 45 | 20 | 30 | 40 | 60 | shard_gpu0_p000_004 |
| vbench10_005 | 0.20 | th_0p20 | 538.963 | 342.273 | 1.575 | 21.166 | 19.340 | 21.620 | 45 | 20 | 30 | 40 | 60 | shard_gpu0_p000_004 |
| vbench10_006 | 0.20 | th_0p20 | 541.959 | 345.262 | 1.570 | 24.488 | 21.920 | 26.030 | 45 | 20 | 30 | 40 | 60 | shard_gpu1_p005_009 |
| vbench10_007 | 0.20 | th_0p20 | 543.609 | 345.175 | 1.575 | 41.091 | 36.920 | 45.600 | 45 | 20 | 30 | 40 | 60 | shard_gpu1_p005_009 |
| vbench10_008 | 0.20 | th_0p20 | 544.246 | 345.589 | 1.575 | 24.702 | 24.060 | 25.530 | 45 | 20 | 30 | 40 | 60 | shard_gpu1_p005_009 |
| vbench10_009 | 0.20 | th_0p20 | 544.039 | 345.194 | 1.576 | 20.576 | 19.200 | 21.480 | 45 | 20 | 30 | 40 | 60 | shard_gpu1_p005_009 |
| vbench10_010 | 0.20 | th_0p20 | 544.254 | 345.598 | 1.575 | 30.007 | 27.170 | 31.240 | 45 | 20 | 30 | 40 | 60 | shard_gpu1_p005_009 |
| vbench10_001 | 0.25 | th_0p25 | 538.211 | 292.264 | 1.842 | 20.771 | 18.330 | 23.360 | 45 | 25 | 25 | 50 | 50 | shard_gpu0_p000_004 |
| vbench10_002 | 0.25 | th_0p25 | 539.263 | 292.193 | 1.846 | 30.568 | 25.600 | 34.930 | 45 | 25 | 25 | 50 | 50 | shard_gpu0_p000_004 |
| vbench10_003 | 0.25 | th_0p25 | 539.507 | 292.478 | 1.845 | 19.502 | 18.230 | 20.610 | 45 | 25 | 25 | 50 | 50 | shard_gpu0_p000_004 |
| vbench10_004 | 0.25 | th_0p25 | 539.041 | 292.498 | 1.843 | 25.246 | 23.710 | 25.960 | 45 | 25 | 25 | 50 | 50 | shard_gpu0_p000_004 |
| vbench10_005 | 0.25 | th_0p25 | 538.963 | 302.394 | 1.782 | 19.745 | 18.710 | 20.240 | 45 | 24 | 26 | 48 | 52 | shard_gpu0_p000_004 |
| vbench10_006 | 0.25 | th_0p25 | 541.959 | 295.182 | 1.836 | 22.907 | 21.450 | 24.390 | 45 | 25 | 25 | 50 | 50 | shard_gpu1_p005_009 |
| vbench10_007 | 0.25 | th_0p25 | 543.609 | 294.672 | 1.845 | 40.594 | 36.650 | 44.760 | 45 | 25 | 25 | 50 | 50 | shard_gpu1_p005_009 |
| vbench10_008 | 0.25 | th_0p25 | 544.246 | 294.602 | 1.847 | 22.794 | 21.810 | 23.810 | 45 | 25 | 25 | 50 | 50 | shard_gpu1_p005_009 |
| vbench10_009 | 0.25 | th_0p25 | 544.039 | 295.105 | 1.844 | 20.612 | 19.460 | 21.470 | 45 | 25 | 25 | 50 | 50 | shard_gpu1_p005_009 |
| vbench10_010 | 0.25 | th_0p25 | 544.254 | 295.050 | 1.845 | 26.950 | 25.320 | 27.550 | 45 | 25 | 25 | 50 | 50 | shard_gpu1_p005_009 |
| vbench10_001 | 0.30 | th_0p30 | 538.211 | 272.089 | 1.978 | 20.156 | 17.510 | 22.890 | 45 | 27 | 23 | 54 | 46 | shard_gpu0_p000_004 |
| vbench10_002 | 0.30 | th_0p30 | 539.263 | 272.075 | 1.982 | 27.949 | 25.020 | 31.030 | 45 | 27 | 23 | 54 | 46 | shard_gpu0_p000_004 |
| vbench10_003 | 0.30 | th_0p30 | 539.507 | 272.710 | 1.978 | 18.548 | 16.940 | 20.400 | 45 | 27 | 23 | 54 | 46 | shard_gpu0_p000_004 |
| vbench10_004 | 0.30 | th_0p30 | 539.041 | 272.811 | 1.976 | 23.604 | 21.880 | 24.440 | 45 | 27 | 23 | 54 | 46 | shard_gpu0_p000_004 |
| vbench10_005 | 0.30 | th_0p30 | 538.963 | 272.531 | 1.978 | 19.056 | 18.130 | 19.530 | 45 | 27 | 23 | 54 | 46 | shard_gpu0_p000_004 |
| vbench10_006 | 0.30 | th_0p30 | 541.959 | 274.333 | 1.976 | 22.090 | 20.400 | 23.100 | 45 | 27 | 23 | 54 | 46 | shard_gpu1_p005_009 |
| vbench10_007 | 0.30 | th_0p30 | 543.609 | 274.542 | 1.980 | 37.090 | 33.000 | 42.840 | 45 | 27 | 23 | 54 | 46 | shard_gpu1_p005_009 |
| vbench10_008 | 0.30 | th_0p30 | 544.246 | 274.529 | 1.982 | 21.400 | 20.580 | 22.440 | 45 | 27 | 23 | 54 | 46 | shard_gpu1_p005_009 |
| vbench10_009 | 0.30 | th_0p30 | 544.039 | 274.846 | 1.979 | 18.104 | 17.560 | 18.640 | 45 | 27 | 23 | 54 | 46 | shard_gpu1_p005_009 |
| vbench10_010 | 0.30 | th_0p30 | 544.254 | 274.679 | 1.981 | 27.157 | 25.450 | 27.830 | 45 | 27 | 23 | 54 | 46 | shard_gpu1_p005_009 |
| vbench10_001 | 0.40 | th_0p40 | 538.211 | 222.220 | 2.422 | 15.556 | 14.410 | 16.780 | 45 | 32 | 18 | 64 | 36 | shard_gpu0_p000_004 |
| vbench10_002 | 0.40 | th_0p40 | 539.263 | 222.271 | 2.426 | 21.129 | 19.650 | 23.530 | 45 | 32 | 18 | 64 | 36 | shard_gpu0_p000_004 |
| vbench10_003 | 0.40 | th_0p40 | 539.507 | 222.313 | 2.427 | 15.406 | 14.400 | 16.530 | 45 | 32 | 18 | 64 | 36 | shard_gpu0_p000_004 |
| vbench10_004 | 0.40 | th_0p40 | 539.041 | 222.709 | 2.420 | 22.424 | 21.410 | 22.950 | 45 | 32 | 18 | 64 | 36 | shard_gpu0_p000_004 |
| vbench10_005 | 0.40 | th_0p40 | 538.963 | 222.481 | 2.423 | 19.037 | 18.130 | 19.530 | 45 | 32 | 18 | 64 | 36 | shard_gpu0_p000_004 |
| vbench10_006 | 0.40 | th_0p40 | 541.959 | 224.129 | 2.418 | 18.743 | 17.290 | 21.160 | 45 | 32 | 18 | 64 | 36 | shard_gpu1_p005_009 |
| vbench10_007 | 0.40 | th_0p40 | 543.609 | 224.262 | 2.424 | 31.681 | 29.120 | 33.880 | 45 | 32 | 18 | 64 | 36 | shard_gpu1_p005_009 |
| vbench10_008 | 0.40 | th_0p40 | 544.246 | 223.984 | 2.430 | 19.900 | 19.140 | 20.510 | 45 | 32 | 18 | 64 | 36 | shard_gpu1_p005_009 |
| vbench10_009 | 0.40 | th_0p40 | 544.039 | 224.494 | 2.423 | 17.428 | 16.890 | 18.120 | 45 | 32 | 18 | 64 | 36 | shard_gpu1_p005_009 |
| vbench10_010 | 0.40 | th_0p40 | 544.254 | 224.253 | 2.427 | 24.709 | 22.690 | 25.830 | 45 | 32 | 18 | 64 | 36 | shard_gpu1_p005_009 |
| vbench10_001 | 0.50 | th_0p50 | 538.211 | 192.516 | 2.796 | 15.582 | 13.690 | 16.770 | 45 | 35 | 15 | 70 | 30 | shard_gpu0_p000_004 |
| vbench10_002 | 0.50 | th_0p50 | 539.263 | 192.328 | 2.804 | 19.977 | 18.770 | 21.640 | 45 | 35 | 15 | 70 | 30 | shard_gpu0_p000_004 |
| vbench10_003 | 0.50 | th_0p50 | 539.507 | 202.406 | 2.665 | 16.490 | 15.290 | 17.790 | 45 | 34 | 16 | 68 | 32 | shard_gpu0_p000_004 |
| vbench10_004 | 0.50 | th_0p50 | 539.041 | 192.684 | 2.798 | 22.051 | 20.950 | 22.520 | 45 | 35 | 15 | 70 | 30 | shard_gpu0_p000_004 |
| vbench10_005 | 0.50 | th_0p50 | 538.963 | 202.444 | 2.662 | 18.471 | 17.800 | 18.930 | 45 | 34 | 16 | 68 | 32 | shard_gpu0_p000_004 |
| vbench10_006 | 0.50 | th_0p50 | 541.959 | 204.093 | 2.655 | 18.148 | 16.760 | 19.420 | 45 | 34 | 16 | 68 | 32 | shard_gpu1_p005_009 |
| vbench10_007 | 0.50 | th_0p50 | 543.609 | 193.962 | 2.803 | 33.678 | 28.840 | 36.450 | 45 | 35 | 15 | 70 | 30 | shard_gpu1_p005_009 |
| vbench10_008 | 0.50 | th_0p50 | 544.246 | 194.047 | 2.805 | 18.941 | 18.030 | 20.040 | 45 | 35 | 15 | 70 | 30 | shard_gpu1_p005_009 |
| vbench10_009 | 0.50 | th_0p50 | 544.039 | 204.179 | 2.665 | 18.963 | 17.800 | 19.810 | 45 | 34 | 16 | 68 | 32 | shard_gpu1_p005_009 |
| vbench10_010 | 0.50 | th_0p50 | 544.254 | 194.021 | 2.805 | 24.487 | 23.160 | 25.250 | 45 | 35 | 15 | 70 | 30 | shard_gpu1_p005_009 |
| vbench10_001 | 0.60 | th_0p60 | 538.211 | 172.331 | 3.123 | 15.752 | 14.160 | 16.870 | 45 | 37 | 13 | 74 | 26 | shard_gpu0_p000_004 |
| vbench10_002 | 0.60 | th_0p60 | 539.263 | 172.496 | 3.126 | 18.535 | 16.980 | 19.920 | 45 | 37 | 13 | 74 | 26 | shard_gpu0_p000_004 |
| vbench10_003 | 0.60 | th_0p60 | 539.507 | 172.542 | 3.127 | 15.567 | 14.410 | 16.260 | 45 | 37 | 13 | 74 | 26 | shard_gpu0_p000_004 |
| vbench10_004 | 0.60 | th_0p60 | 539.041 | 172.667 | 3.122 | 21.856 | 20.750 | 22.350 | 45 | 37 | 13 | 74 | 26 | shard_gpu0_p000_004 |
| vbench10_005 | 0.60 | th_0p60 | 538.963 | 172.491 | 3.125 | 18.521 | 17.800 | 19.030 | 45 | 37 | 13 | 74 | 26 | shard_gpu0_p000_004 |
| vbench10_006 | 0.60 | th_0p60 | 541.959 | 173.651 | 3.121 | 17.266 | 14.530 | 19.460 | 45 | 37 | 13 | 74 | 26 | shard_gpu1_p005_009 |
| vbench10_007 | 0.60 | th_0p60 | 543.609 | 173.935 | 3.125 | 34.239 | 31.400 | 37.980 | 45 | 37 | 13 | 74 | 26 | shard_gpu1_p005_009 |
| vbench10_008 | 0.60 | th_0p60 | 544.246 | 173.816 | 3.131 | 21.054 | 20.150 | 21.650 | 45 | 37 | 13 | 74 | 26 | shard_gpu1_p005_009 |
| vbench10_009 | 0.60 | th_0p60 | 544.039 | 174.257 | 3.122 | 19.065 | 18.060 | 20.210 | 45 | 37 | 13 | 74 | 26 | shard_gpu1_p005_009 |
| vbench10_010 | 0.60 | th_0p60 | 544.254 | 173.738 | 3.133 | 24.734 | 23.840 | 25.120 | 45 | 37 | 13 | 74 | 26 | shard_gpu1_p005_009 |
| vbench10_001 | 0.70 | th_0p70 | 538.211 | 152.604 | 3.527 | 16.205 | 14.460 | 17.470 | 45 | 39 | 11 | 78 | 22 | shard_gpu0_p000_004 |
| vbench10_002 | 0.70 | th_0p70 | 539.263 | 162.372 | 3.321 | 18.266 | 16.750 | 19.430 | 45 | 38 | 12 | 76 | 24 | shard_gpu0_p000_004 |
| vbench10_003 | 0.70 | th_0p70 | 539.507 | 162.590 | 3.318 | 15.797 | 14.690 | 16.750 | 45 | 38 | 12 | 76 | 24 | shard_gpu0_p000_004 |
| vbench10_004 | 0.70 | th_0p70 | 539.041 | 162.638 | 3.314 | 21.960 | 21.150 | 22.320 | 45 | 38 | 12 | 76 | 24 | shard_gpu0_p000_004 |
| vbench10_005 | 0.70 | th_0p70 | 538.963 | 162.482 | 3.317 | 17.392 | 16.710 | 17.660 | 45 | 38 | 12 | 76 | 24 | shard_gpu0_p000_004 |
| vbench10_006 | 0.70 | th_0p70 | 541.959 | 163.712 | 3.310 | 16.752 | 14.500 | 19.210 | 45 | 38 | 12 | 76 | 24 | shard_gpu1_p005_009 |
| vbench10_007 | 0.70 | th_0p70 | 543.609 | 163.804 | 3.319 | 30.390 | 22.520 | 33.980 | 45 | 38 | 12 | 76 | 24 | shard_gpu1_p005_009 |
| vbench10_008 | 0.70 | th_0p70 | 544.246 | 163.679 | 3.325 | 20.366 | 19.520 | 20.890 | 45 | 38 | 12 | 76 | 24 | shard_gpu1_p005_009 |
| vbench10_009 | 0.70 | th_0p70 | 544.039 | 163.745 | 3.322 | 17.714 | 16.510 | 18.430 | 45 | 38 | 12 | 76 | 24 | shard_gpu1_p005_009 |
| vbench10_010 | 0.70 | th_0p70 | 544.254 | 163.497 | 3.329 | 23.539 | 21.960 | 24.160 | 45 | 38 | 12 | 76 | 24 | shard_gpu1_p005_009 |
| vbench10_001 | 0.80 | th_0p80 | 538.211 | 152.498 | 3.529 | 16.150 | 14.200 | 17.480 | 45 | 39 | 11 | 78 | 22 | shard_gpu0_p000_004 |
| vbench10_002 | 0.80 | th_0p80 | 539.263 | 152.519 | 3.536 | 17.732 | 16.400 | 18.970 | 45 | 39 | 11 | 78 | 22 | shard_gpu0_p000_004 |
| vbench10_003 | 0.80 | th_0p80 | 539.507 | 152.557 | 3.536 | 15.735 | 14.610 | 16.650 | 45 | 39 | 11 | 78 | 22 | shard_gpu0_p000_004 |
| vbench10_004 | 0.80 | th_0p80 | 539.041 | 152.575 | 3.533 | 21.195 | 20.430 | 21.550 | 45 | 39 | 11 | 78 | 22 | shard_gpu0_p000_004 |
| vbench10_005 | 0.80 | th_0p80 | 538.963 | 152.567 | 3.533 | 17.447 | 17.050 | 17.630 | 45 | 39 | 11 | 78 | 22 | shard_gpu0_p000_004 |
| vbench10_006 | 0.80 | th_0p80 | 541.959 | 153.662 | 3.527 | 16.992 | 14.650 | 19.250 | 45 | 39 | 11 | 78 | 22 | shard_gpu1_p005_009 |
| vbench10_007 | 0.80 | th_0p80 | 543.609 | 153.834 | 3.534 | 29.653 | 22.170 | 32.330 | 45 | 39 | 11 | 78 | 22 | shard_gpu1_p005_009 |
| vbench10_008 | 0.80 | th_0p80 | 544.246 | 153.624 | 3.543 | 20.699 | 20.050 | 21.170 | 45 | 39 | 11 | 78 | 22 | shard_gpu1_p005_009 |
| vbench10_009 | 0.80 | th_0p80 | 544.039 | 153.869 | 3.536 | 17.151 | 16.240 | 17.670 | 45 | 39 | 11 | 78 | 22 | shard_gpu1_p005_009 |
| vbench10_010 | 0.80 | th_0p80 | 544.254 | 153.537 | 3.545 | 23.821 | 22.910 | 24.270 | 45 | 39 | 11 | 78 | 22 | shard_gpu1_p005_009 |


## 3. 10-Prompt Aggregate Table

One row per threshold, aggregated across all 10 prompts.

| label | threshold | n | base_total_s | seacache_total_s | overall_speedup | mean_psnr | min_psnr | reuse | recomp | reuse_branch | recomp_branch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| th_0p10 | 0.10 | 10 | 5413.092 | 4886.090 | 1.108 | 35.897 | 27.970 | 56 | 444 | 112 | 888 |
| th_0p15 | 0.15 | 10 | 5413.092 | 3838.109 | 1.410 | 28.001 | 18.520 | 160 | 340 | 320 | 680 |
| th_0p20 | 0.20 | 10 | 5413.092 | 3438.230 | 1.574 | 25.958 | 17.530 | 200 | 300 | 400 | 600 |
| th_0p25 | 0.25 | 10 | 5413.092 | 2946.438 | 1.837 | 24.969 | 18.230 | 249 | 251 | 498 | 502 |
| th_0p30 | 0.30 | 10 | 5413.092 | 2735.145 | 1.979 | 23.515 | 16.940 | 270 | 230 | 540 | 460 |
| th_0p40 | 0.40 | 10 | 5413.092 | 2233.116 | 2.424 | 20.601 | 14.400 | 320 | 180 | 640 | 360 |
| th_0p50 | 0.50 | 10 | 5413.092 | 1972.680 | 2.744 | 20.679 | 13.690 | 346 | 154 | 692 | 308 |
| th_0p60 | 0.60 | 10 | 5413.092 | 1731.924 | 3.125 | 20.659 | 14.160 | 370 | 130 | 740 | 260 |
| th_0p70 | 0.70 | 10 | 5413.092 | 1621.123 | 3.339 | 19.838 | 14.460 | 381 | 119 | 762 | 238 |
| th_0p80 | 0.80 | 10 | 5413.092 | 1531.242 | 3.535 | 19.657 | 14.200 | 390 | 110 | 780 | 220 |


## 4. Simple Analysis

- Fastest threshold: `th_0p80` with overall speedup `3.535` and mean PSNR `19.657`.

- Highest mean PSNR: `th_0p10` with speedup `1.108` and mean PSNR `35.897`.

- Best quality with at least 1.5x speedup: `th_0p20` with speedup `1.574` and mean PSNR `25.958`.

- Best quality with at least 2x speedup: `th_0p50` with speedup `2.744` and mean PSNR `20.679`.

- Timestep-only SeaCache has a smooth threshold-speed tradeoff: reuse increases from threshold 0.10 to 0.80, speedup rises from about 1.11x to 3.54x, and mean PSNR falls from about 35.90 to 19.66.

- For quality-sensitive use, threshold 0.10 is the conservative point. Thresholds around 0.20-0.30 are middle-ground settings. Thresholds 0.50 and above mainly optimize speed and should be treated as low-quality/aggressive settings on this prompt set.
