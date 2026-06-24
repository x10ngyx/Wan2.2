# VBench10 Three-Cache SEA Merge Report

Generated: 2026-06-23 Asia/Shanghai.

## 1. Experiment Configuration

This experiment evaluates the SEA-style merged cache method on the 10-prompt VBench10 subset. It combines timestep SeaCache, block-group SEA cache, and CFG SEA cache. The threshold grid is `4 x 4 x 4 = 64` candidates per prompt. Two GPU shards each cover five prompts; the tables below use the merged 10-prompt result.

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
| sample_guide_scale | [3.0, 4.0] |
| python_bin | /hy-tmp/env/Wan2.2/bin/python |
| ffprobe_bin | /hy-tmp/env/Wan2.2/bin/ffprobe |
| offload_model | True |
| convert_model_dtype | True |
| resume_existing | True |
| timestep_cache | seacache |
| block_cache | block-group sea_full_rel_l1 accumulated |
| cfg_cache | sea-threshold |
| seacache_num_steps | None |
| seacache_use_ret_steps | False |
| seacache_power_exp | 3.0 |
| seacache_power_const | 1.0 |
| seacache_eps | 1e-16 |
| seacache_norm_mode | mean |
| timestep_thresholds | 0.05 0.10 0.20 0.50 |
| block_thresholds | 0.05 0.10 0.20 0.50 |
| cfg_thresholds | 0.05 0.10 0.20 0.50 |
| candidate_count_per_prompt | 64 |
| block_group_size | 5 |
| block_group_max_reuse | 50 |
| block_group_eps | 1e-06 |
| block_group_ret_steps | 1 |
| block_group_cutoff_steps | 1 |
| block_group_sea_power_exp | 3.0 |
| block_group_sea_power_const | 1.0 |
| block_group_sea_norm_mode | mean |
| cfg_eps | 1e-06 |
| cfg_sea_power_exp | 3.0 |
| cfg_sea_power_const | 1.0 |
| cfg_sea_norm_mode | mean |
| cfg_ret_steps | 1 |
| cfg_cutoff_steps | 1 |
| cache_order | CFG outer; branch timestep cache; block-group cache only on timestep miss |
| per_prompt_result_dirs | True |
| shard_gpu0 | prompt_start=0, prompt_limit=5, selected_prompt_count=5, root=/hy-tmp/wan22_three_cache_sea_vbench10_50step_45f_480p_20260619_3sea_gpu0_p000_004 |
| shard_gpu1 | prompt_start=5, prompt_limit=5, selected_prompt_count=5, root=/hy-tmp/wan22_three_cache_sea_vbench10_50step_45f_480p_20260619_3sea_gpu1_p005_009 |


Source result files:

- full per-prompt table: `/hy-tmp/wan22_three_cache_sea_vbench10_50step_45f_480p_20260619_3sea_merged_parent/merged/summary.csv`

- 10-prompt aggregate table: `/hy-tmp/wan22_three_cache_sea_vbench10_50step_45f_480p_20260619_3sea_merged_parent/merged/aggregate_by_candidate.csv`


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

One row per `(prompt, threshold combination)`. Timing columns are seconds; `speedup = baseline_elapsed / candidate_elapsed`; larger `mean_psnr` is closer to the no-cache baseline.

| sample_id | candidate | ts_th | bg_th | cfg_th | base_s | cand_s | speedup | mean_psnr | min_psnr | max_psnr | psnr_frames | t_reuse | t_recomp | bg_reuse | bg_recomp | cfg_reuse | cfg_recomp | shard |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vbench10_001 | sea_ts_0p05__sea_bg_0p05__sea_cfg_0p05 | 0.05 | 0.05 | 0.05 | 537.977 | 581.691 | 0.925 |  |  |  | 0 | 0 | 50 | 0 | 800 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p05__sea_bg_0p05__sea_cfg_0p05 | 0.05 | 0.05 | 0.05 | 538.450 | 582.137 | 0.925 |  |  |  | 0 | 0 | 50 | 0 | 800 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p05__sea_bg_0p05__sea_cfg_0p05 | 0.05 | 0.05 | 0.05 | 539.052 | 582.374 | 0.926 |  |  |  | 0 | 0 | 50 | 0 | 800 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p05__sea_bg_0p05__sea_cfg_0p05 | 0.05 | 0.05 | 0.05 | 538.562 | 582.376 | 0.925 |  |  |  | 0 | 0 | 50 | 0 | 800 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p05__sea_bg_0p05__sea_cfg_0p05 | 0.05 | 0.05 | 0.05 | 539.330 | 582.681 | 0.926 |  |  |  | 0 | 0 | 50 | 0 | 800 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p05__sea_bg_0p05__sea_cfg_0p05 | 0.05 | 0.05 | 0.05 | 542.662 | 586.899 | 0.925 |  |  |  | 0 | 0 | 50 | 0 | 800 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p05__sea_bg_0p05__sea_cfg_0p05 | 0.05 | 0.05 | 0.05 | 544.363 | 588.415 | 0.925 |  |  |  | 0 | 0 | 50 | 0 | 800 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p05__sea_bg_0p05__sea_cfg_0p05 | 0.05 | 0.05 | 0.05 | 544.006 | 587.957 | 0.925 |  |  |  | 0 | 0 | 50 | 0 | 800 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p05__sea_bg_0p05__sea_cfg_0p05 | 0.05 | 0.05 | 0.05 | 543.652 | 587.917 | 0.925 |  |  |  | 0 | 0 | 50 | 0 | 800 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p05__sea_bg_0p05__sea_cfg_0p05 | 0.05 | 0.05 | 0.05 | 544.915 | 588.597 | 0.926 |  |  |  | 0 | 0 | 50 | 0 | 800 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p05__sea_bg_0p05__sea_cfg_0p10 | 0.05 | 0.05 | 0.10 | 537.977 | 548.979 | 0.980 | 34.483 | 31.690 | 37.130 | 45 | 0 | 50 | 0 | 752 | 6 | 44 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p05__sea_bg_0p05__sea_cfg_0p10 | 0.05 | 0.05 | 0.10 | 538.450 | 548.452 | 0.982 | 44.303 | 43.200 | 45.210 | 45 | 0 | 50 | 0 | 752 | 6 | 44 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p05__sea_bg_0p05__sea_cfg_0p10 | 0.05 | 0.05 | 0.10 | 539.052 | 554.532 | 0.972 | 32.594 | 29.440 | 34.670 | 45 | 0 | 50 | 0 | 760 | 5 | 45 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p05__sea_bg_0p05__sea_cfg_0p10 | 0.05 | 0.05 | 0.10 | 538.562 | 548.621 | 0.982 | 38.685 | 35.460 | 40.140 | 45 | 0 | 50 | 0 | 752 | 6 | 44 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p05__sea_bg_0p05__sea_cfg_0p10 | 0.05 | 0.05 | 0.10 | 539.330 | 554.868 | 0.972 | 30.669 | 29.110 | 31.340 | 45 | 0 | 50 | 0 | 760 | 5 | 45 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p05__sea_bg_0p05__sea_cfg_0p10 | 0.05 | 0.05 | 0.10 | 542.662 | 558.974 | 0.971 | 36.624 | 33.870 | 39.510 | 45 | 0 | 50 | 0 | 760 | 5 | 45 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p05__sea_bg_0p05__sea_cfg_0p10 | 0.05 | 0.05 | 0.10 | 544.363 | 554.386 | 0.982 | 48.582 | 45.490 | 50.930 | 45 | 0 | 50 | 0 | 752 | 6 | 44 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p05__sea_bg_0p05__sea_cfg_0p10 | 0.05 | 0.05 | 0.10 | 544.006 | 554.009 | 0.982 | 33.297 | 32.890 | 34.040 | 45 | 0 | 50 | 0 | 752 | 6 | 44 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p05__sea_bg_0p05__sea_cfg_0p10 | 0.05 | 0.05 | 0.10 | 543.652 | 559.584 | 0.972 | 33.807 | 32.360 | 35.290 | 45 | 0 | 50 | 0 | 760 | 5 | 45 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p05__sea_bg_0p05__sea_cfg_0p10 | 0.05 | 0.05 | 0.10 | 544.915 | 553.516 | 0.984 | 41.947 | 39.800 | 42.770 | 45 | 0 | 50 | 0 | 752 | 6 | 44 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p05__sea_bg_0p05__sea_cfg_0p20 | 0.05 | 0.05 | 0.20 | 537.977 | 470.883 | 1.142 | 18.996 | 16.350 | 22.130 | 45 | 0 | 50 | 0 | 640 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p05__sea_bg_0p05__sea_cfg_0p20 | 0.05 | 0.05 | 0.20 | 538.450 | 470.110 | 1.145 | 32.943 | 25.580 | 38.050 | 45 | 0 | 50 | 0 | 640 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p05__sea_bg_0p05__sea_cfg_0p20 | 0.05 | 0.05 | 0.20 | 539.052 | 470.041 | 1.147 | 21.665 | 19.370 | 24.520 | 45 | 0 | 50 | 0 | 640 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p05__sea_bg_0p05__sea_cfg_0p20 | 0.05 | 0.05 | 0.20 | 538.562 | 470.133 | 1.146 | 25.385 | 24.580 | 25.710 | 45 | 0 | 50 | 0 | 640 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p05__sea_bg_0p05__sea_cfg_0p20 | 0.05 | 0.05 | 0.20 | 539.330 | 470.185 | 1.147 | 22.338 | 20.900 | 22.850 | 45 | 0 | 50 | 0 | 640 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p05__sea_bg_0p05__sea_cfg_0p20 | 0.05 | 0.05 | 0.20 | 542.662 | 474.397 | 1.144 | 25.524 | 23.960 | 28.410 | 45 | 0 | 50 | 0 | 640 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p05__sea_bg_0p05__sea_cfg_0p20 | 0.05 | 0.05 | 0.20 | 544.363 | 475.042 | 1.146 | 42.347 | 37.360 | 47.590 | 45 | 0 | 50 | 0 | 640 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p05__sea_bg_0p05__sea_cfg_0p20 | 0.05 | 0.05 | 0.20 | 544.006 | 474.604 | 1.146 | 25.636 | 24.080 | 26.630 | 45 | 0 | 50 | 0 | 640 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p05__sea_bg_0p05__sea_cfg_0p20 | 0.05 | 0.05 | 0.20 | 543.652 | 474.443 | 1.146 | 20.200 | 19.020 | 20.900 | 45 | 0 | 50 | 0 | 640 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p05__sea_bg_0p05__sea_cfg_0p20 | 0.05 | 0.05 | 0.20 | 544.915 | 473.548 | 1.151 | 30.349 | 26.900 | 31.730 | 45 | 0 | 50 | 0 | 640 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p05__sea_bg_0p05__sea_cfg_0p50 | 0.05 | 0.05 | 0.50 | 537.977 | 380.981 | 1.412 | 14.735 | 13.020 | 16.170 | 45 | 0 | 50 | 0 | 512 | 36 | 14 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p05__sea_bg_0p05__sea_cfg_0p50 | 0.05 | 0.05 | 0.50 | 538.450 | 380.324 | 1.416 | 20.173 | 18.150 | 23.060 | 45 | 0 | 50 | 0 | 512 | 36 | 14 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p05__sea_bg_0p05__sea_cfg_0p50 | 0.05 | 0.05 | 0.50 | 539.052 | 385.847 | 1.397 | 15.077 | 14.160 | 15.810 | 45 | 0 | 50 | 0 | 520 | 35 | 15 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p05__sea_bg_0p05__sea_cfg_0p50 | 0.05 | 0.05 | 0.50 | 538.562 | 380.211 | 1.416 | 24.200 | 23.140 | 25.100 | 45 | 0 | 50 | 0 | 512 | 36 | 14 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p05__sea_bg_0p05__sea_cfg_0p50 | 0.05 | 0.05 | 0.50 | 539.330 | 386.034 | 1.397 | 18.076 | 17.240 | 18.540 | 45 | 0 | 50 | 0 | 520 | 35 | 15 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p05__sea_bg_0p05__sea_cfg_0p50 | 0.05 | 0.05 | 0.50 | 542.662 | 389.223 | 1.394 | 16.850 | 14.900 | 19.330 | 45 | 0 | 50 | 0 | 520 | 35 | 15 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p05__sea_bg_0p05__sea_cfg_0p50 | 0.05 | 0.05 | 0.50 | 544.363 | 384.214 | 1.417 | 32.407 | 22.920 | 36.940 | 45 | 0 | 50 | 0 | 512 | 36 | 14 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p05__sea_bg_0p05__sea_cfg_0p50 | 0.05 | 0.05 | 0.50 | 544.006 | 383.802 | 1.417 | 20.570 | 19.790 | 21.320 | 45 | 0 | 50 | 0 | 512 | 36 | 14 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p05__sea_bg_0p05__sea_cfg_0p50 | 0.05 | 0.05 | 0.50 | 543.652 | 389.324 | 1.396 | 19.238 | 18.580 | 19.840 | 45 | 0 | 50 | 0 | 520 | 35 | 15 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p05__sea_bg_0p05__sea_cfg_0p50 | 0.05 | 0.05 | 0.50 | 544.915 | 383.254 | 1.422 | 25.078 | 23.710 | 25.770 | 45 | 0 | 50 | 0 | 512 | 36 | 14 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p05__sea_bg_0p10__sea_cfg_0p05 | 0.05 | 0.10 | 0.05 | 537.977 | 531.399 | 1.012 | 33.104 | 28.370 | 37.880 | 45 | 0 | 50 | 82 | 718 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p05__sea_bg_0p10__sea_cfg_0p05 | 0.05 | 0.10 | 0.05 | 538.450 | 531.549 | 1.013 | 42.681 | 40.730 | 43.580 | 45 | 0 | 50 | 82 | 718 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p05__sea_bg_0p10__sea_cfg_0p05 | 0.05 | 0.10 | 0.05 | 539.052 | 541.194 | 0.996 | 32.027 | 29.930 | 33.950 | 45 | 0 | 50 | 66 | 734 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p05__sea_bg_0p10__sea_cfg_0p05 | 0.05 | 0.10 | 0.05 | 538.562 | 531.434 | 1.013 | 36.586 | 33.570 | 37.980 | 45 | 0 | 50 | 82 | 718 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p05__sea_bg_0p10__sea_cfg_0p05 | 0.05 | 0.10 | 0.05 | 539.330 | 541.780 | 0.995 | 30.440 | 28.360 | 31.530 | 45 | 0 | 50 | 66 | 734 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p05__sea_bg_0p10__sea_cfg_0p05 | 0.05 | 0.10 | 0.05 | 542.662 | 545.464 | 0.995 | 35.233 | 33.320 | 38.230 | 45 | 0 | 50 | 66 | 734 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p05__sea_bg_0p10__sea_cfg_0p05 | 0.05 | 0.10 | 0.05 | 544.363 | 545.272 | 0.998 | 46.750 | 44.700 | 49.200 | 45 | 0 | 50 | 68 | 732 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p05__sea_bg_0p10__sea_cfg_0p05 | 0.05 | 0.10 | 0.05 | 544.006 | 536.432 | 1.014 | 31.005 | 30.040 | 32.140 | 45 | 0 | 50 | 82 | 718 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p05__sea_bg_0p10__sea_cfg_0p05 | 0.05 | 0.10 | 0.05 | 543.652 | 546.442 | 0.995 | 32.757 | 31.070 | 34.060 | 45 | 0 | 50 | 66 | 734 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p05__sea_bg_0p10__sea_cfg_0p05 | 0.05 | 0.10 | 0.05 | 544.915 | 539.937 | 1.009 | 40.206 | 38.170 | 41.310 | 45 | 0 | 50 | 75 | 725 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p05__sea_bg_0p10__sea_cfg_0p10 | 0.05 | 0.10 | 0.10 | 537.977 | 523.658 | 1.027 | 32.716 | 28.270 | 36.890 | 45 | 0 | 50 | 41 | 711 | 6 | 44 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p05__sea_bg_0p10__sea_cfg_0p10 | 0.05 | 0.10 | 0.10 | 538.450 | 523.473 | 1.029 | 42.664 | 40.730 | 43.830 | 45 | 0 | 50 | 41 | 711 | 6 | 44 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p05__sea_bg_0p10__sea_cfg_0p10 | 0.05 | 0.10 | 0.10 | 539.052 | 534.334 | 1.009 | 31.770 | 28.320 | 34.030 | 45 | 0 | 50 | 33 | 727 | 5 | 45 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p05__sea_bg_0p10__sea_cfg_0p10 | 0.05 | 0.10 | 0.10 | 538.562 | 523.273 | 1.029 | 36.512 | 33.400 | 37.840 | 45 | 0 | 50 | 41 | 711 | 6 | 44 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p05__sea_bg_0p10__sea_cfg_0p10 | 0.05 | 0.10 | 0.10 | 539.330 | 534.253 | 1.010 | 30.144 | 28.510 | 31.030 | 45 | 0 | 50 | 33 | 727 | 5 | 45 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p05__sea_bg_0p10__sea_cfg_0p10 | 0.05 | 0.10 | 0.10 | 542.662 | 538.568 | 1.008 | 35.322 | 33.040 | 38.530 | 45 | 0 | 50 | 33 | 727 | 5 | 45 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p05__sea_bg_0p10__sea_cfg_0p10 | 0.05 | 0.10 | 0.10 | 544.363 | 532.910 | 1.021 | 47.912 | 45.270 | 50.540 | 45 | 0 | 50 | 34 | 718 | 6 | 44 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p05__sea_bg_0p10__sea_cfg_0p10 | 0.05 | 0.10 | 0.10 | 544.006 | 528.446 | 1.029 | 31.423 | 30.810 | 32.190 | 45 | 0 | 50 | 41 | 711 | 6 | 44 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p05__sea_bg_0p10__sea_cfg_0p10 | 0.05 | 0.10 | 0.10 | 543.652 | 538.710 | 1.009 | 32.796 | 31.380 | 33.960 | 45 | 0 | 50 | 33 | 727 | 5 | 45 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p05__sea_bg_0p10__sea_cfg_0p10 | 0.05 | 0.10 | 0.10 | 544.915 | 531.713 | 1.025 | 40.006 | 38.120 | 41.080 | 45 | 0 | 50 | 34 | 718 | 6 | 44 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p05__sea_bg_0p10__sea_cfg_0p20 | 0.05 | 0.10 | 0.20 | 537.977 | 444.936 | 1.209 | 19.127 | 16.460 | 22.150 | 45 | 0 | 50 | 41 | 599 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p05__sea_bg_0p10__sea_cfg_0p20 | 0.05 | 0.10 | 0.20 | 538.450 | 444.911 | 1.210 | 32.158 | 25.450 | 36.170 | 45 | 0 | 50 | 41 | 599 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p05__sea_bg_0p10__sea_cfg_0p20 | 0.05 | 0.10 | 0.20 | 539.052 | 450.255 | 1.197 | 21.796 | 19.400 | 24.840 | 45 | 0 | 50 | 33 | 607 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p05__sea_bg_0p10__sea_cfg_0p20 | 0.05 | 0.10 | 0.20 | 538.562 | 444.801 | 1.211 | 25.660 | 24.840 | 26.060 | 45 | 0 | 50 | 41 | 599 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p05__sea_bg_0p10__sea_cfg_0p20 | 0.05 | 0.10 | 0.20 | 539.330 | 449.782 | 1.199 | 22.478 | 21.060 | 23.040 | 45 | 0 | 50 | 33 | 607 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p05__sea_bg_0p10__sea_cfg_0p20 | 0.05 | 0.10 | 0.20 | 542.662 | 453.538 | 1.197 | 25.731 | 24.070 | 28.490 | 45 | 0 | 50 | 33 | 607 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p05__sea_bg_0p10__sea_cfg_0p20 | 0.05 | 0.10 | 0.20 | 544.363 | 452.754 | 1.202 | 41.435 | 37.380 | 46.310 | 45 | 0 | 50 | 34 | 606 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p05__sea_bg_0p10__sea_cfg_0p20 | 0.05 | 0.10 | 0.20 | 544.006 | 449.264 | 1.211 | 24.974 | 23.790 | 25.760 | 45 | 0 | 50 | 41 | 599 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p05__sea_bg_0p10__sea_cfg_0p20 | 0.05 | 0.10 | 0.20 | 543.652 | 453.708 | 1.198 | 20.318 | 19.090 | 21.080 | 45 | 0 | 50 | 33 | 607 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p05__sea_bg_0p10__sea_cfg_0p20 | 0.05 | 0.10 | 0.20 | 544.915 | 452.411 | 1.204 | 30.143 | 27.140 | 31.470 | 45 | 0 | 50 | 34 | 606 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p05__sea_bg_0p10__sea_cfg_0p50 | 0.05 | 0.10 | 0.50 | 537.977 | 355.395 | 1.514 | 14.868 | 13.100 | 16.310 | 45 | 0 | 50 | 41 | 471 | 36 | 14 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p05__sea_bg_0p10__sea_cfg_0p50 | 0.05 | 0.10 | 0.50 | 538.450 | 355.380 | 1.515 | 20.266 | 18.200 | 23.090 | 45 | 0 | 50 | 41 | 471 | 36 | 14 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p05__sea_bg_0p10__sea_cfg_0p50 | 0.05 | 0.10 | 0.50 | 539.052 | 365.680 | 1.474 | 15.103 | 14.180 | 15.910 | 45 | 0 | 50 | 33 | 487 | 35 | 15 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p05__sea_bg_0p10__sea_cfg_0p50 | 0.05 | 0.10 | 0.50 | 538.562 | 355.285 | 1.516 | 24.340 | 23.160 | 25.150 | 45 | 0 | 50 | 41 | 471 | 36 | 14 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p05__sea_bg_0p10__sea_cfg_0p50 | 0.05 | 0.10 | 0.50 | 539.330 | 365.875 | 1.474 | 18.156 | 17.330 | 18.630 | 45 | 0 | 50 | 33 | 487 | 35 | 15 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p05__sea_bg_0p10__sea_cfg_0p50 | 0.05 | 0.10 | 0.50 | 542.662 | 368.740 | 1.472 | 16.882 | 14.910 | 19.340 | 45 | 0 | 50 | 33 | 487 | 35 | 15 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p05__sea_bg_0p10__sea_cfg_0p50 | 0.05 | 0.10 | 0.50 | 544.363 | 361.863 | 1.504 | 32.639 | 22.870 | 38.000 | 45 | 0 | 50 | 34 | 478 | 36 | 14 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p05__sea_bg_0p10__sea_cfg_0p50 | 0.05 | 0.10 | 0.50 | 544.006 | 358.130 | 1.519 | 20.910 | 20.110 | 21.620 | 45 | 0 | 50 | 41 | 471 | 36 | 14 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p05__sea_bg_0p10__sea_cfg_0p50 | 0.05 | 0.10 | 0.50 | 543.652 | 368.764 | 1.474 | 19.253 | 18.550 | 19.910 | 45 | 0 | 50 | 33 | 487 | 35 | 15 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p05__sea_bg_0p10__sea_cfg_0p50 | 0.05 | 0.10 | 0.50 | 544.915 | 361.872 | 1.506 | 25.345 | 24.140 | 26.100 | 45 | 0 | 50 | 34 | 478 | 36 | 14 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p05__sea_bg_0p20__sea_cfg_0p05 | 0.05 | 0.20 | 0.05 | 537.977 | 391.693 | 1.373 | 20.476 | 17.620 | 23.370 | 45 | 0 | 50 | 306 | 494 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p05__sea_bg_0p20__sea_cfg_0p05 | 0.05 | 0.20 | 0.05 | 538.450 | 391.606 | 1.375 | 32.275 | 25.460 | 37.240 | 45 | 0 | 50 | 306 | 494 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p05__sea_bg_0p20__sea_cfg_0p05 | 0.05 | 0.20 | 0.05 | 539.052 | 391.618 | 1.376 | 20.204 | 17.480 | 23.420 | 45 | 0 | 50 | 306 | 494 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p05__sea_bg_0p20__sea_cfg_0p05 | 0.05 | 0.20 | 0.05 | 538.562 | 391.486 | 1.376 | 24.977 | 24.070 | 25.640 | 45 | 0 | 50 | 306 | 494 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p05__sea_bg_0p20__sea_cfg_0p05 | 0.05 | 0.20 | 0.05 | 539.330 | 395.452 | 1.364 | 21.165 | 19.950 | 21.630 | 45 | 0 | 50 | 300 | 500 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p05__sea_bg_0p20__sea_cfg_0p05 | 0.05 | 0.20 | 0.05 | 542.662 | 394.643 | 1.375 | 24.555 | 22.090 | 26.380 | 45 | 0 | 50 | 306 | 494 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p05__sea_bg_0p20__sea_cfg_0p05 | 0.05 | 0.20 | 0.05 | 544.363 | 394.284 | 1.381 | 41.380 | 37.480 | 45.920 | 45 | 0 | 50 | 306 | 494 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p05__sea_bg_0p20__sea_cfg_0p05 | 0.05 | 0.20 | 0.05 | 544.006 | 394.870 | 1.378 | 24.076 | 23.510 | 24.800 | 45 | 0 | 50 | 306 | 494 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p05__sea_bg_0p20__sea_cfg_0p05 | 0.05 | 0.20 | 0.05 | 543.652 | 394.314 | 1.379 | 20.494 | 19.150 | 21.420 | 45 | 0 | 50 | 306 | 494 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p05__sea_bg_0p20__sea_cfg_0p05 | 0.05 | 0.20 | 0.05 | 544.915 | 397.506 | 1.371 | 26.354 | 25.150 | 27.030 | 45 | 0 | 50 | 301 | 499 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p05__sea_bg_0p20__sea_cfg_0p10 | 0.05 | 0.20 | 0.10 | 537.977 | 378.564 | 1.421 | 20.535 | 17.660 | 23.560 | 45 | 0 | 50 | 273 | 479 | 6 | 44 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p05__sea_bg_0p20__sea_cfg_0p10 | 0.05 | 0.20 | 0.10 | 538.450 | 378.061 | 1.424 | 31.087 | 25.160 | 34.630 | 45 | 0 | 50 | 273 | 479 | 6 | 44 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p05__sea_bg_0p20__sea_cfg_0p10 | 0.05 | 0.20 | 0.10 | 539.052 | 378.783 | 1.423 | 19.935 | 17.390 | 22.850 | 45 | 0 | 50 | 273 | 479 | 6 | 44 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p05__sea_bg_0p20__sea_cfg_0p10 | 0.05 | 0.20 | 0.10 | 538.562 | 378.485 | 1.423 | 24.902 | 24.110 | 25.660 | 45 | 0 | 50 | 273 | 479 | 6 | 44 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p05__sea_bg_0p20__sea_cfg_0p10 | 0.05 | 0.20 | 0.10 | 539.330 | 381.460 | 1.414 | 20.909 | 19.740 | 21.380 | 45 | 0 | 50 | 267 | 485 | 6 | 44 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p05__sea_bg_0p20__sea_cfg_0p10 | 0.05 | 0.20 | 0.10 | 542.662 | 382.588 | 1.418 | 24.385 | 21.990 | 25.890 | 45 | 0 | 50 | 281 | 479 | 5 | 45 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p05__sea_bg_0p20__sea_cfg_0p10 | 0.05 | 0.20 | 0.10 | 544.363 | 385.574 | 1.412 | 41.230 | 37.870 | 45.650 | 45 | 0 | 50 | 266 | 486 | 6 | 44 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p05__sea_bg_0p20__sea_cfg_0p10 | 0.05 | 0.20 | 0.10 | 544.006 | 381.591 | 1.426 | 23.569 | 23.100 | 24.140 | 45 | 0 | 50 | 273 | 479 | 6 | 44 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p05__sea_bg_0p20__sea_cfg_0p10 | 0.05 | 0.20 | 0.10 | 543.652 | 381.607 | 1.425 | 20.396 | 19.050 | 21.290 | 45 | 0 | 50 | 273 | 479 | 6 | 44 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p05__sea_bg_0p20__sea_cfg_0p10 | 0.05 | 0.20 | 0.10 | 544.915 | 384.383 | 1.418 | 26.407 | 25.220 | 27.060 | 45 | 0 | 50 | 268 | 484 | 6 | 44 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p05__sea_bg_0p20__sea_cfg_0p20 | 0.05 | 0.20 | 0.20 | 537.977 | 372.617 | 1.444 | 19.757 | 17.120 | 22.660 | 45 | 0 | 50 | 156 | 484 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p05__sea_bg_0p20__sea_cfg_0p20 | 0.05 | 0.20 | 0.20 | 538.450 | 373.800 | 1.440 | 32.920 | 27.990 | 37.090 | 45 | 0 | 50 | 155 | 485 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p05__sea_bg_0p20__sea_cfg_0p20 | 0.05 | 0.20 | 0.20 | 539.052 | 373.763 | 1.442 | 20.083 | 17.570 | 23.090 | 45 | 0 | 50 | 155 | 485 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p05__sea_bg_0p20__sea_cfg_0p20 | 0.05 | 0.20 | 0.20 | 538.562 | 373.409 | 1.442 | 24.844 | 24.000 | 25.190 | 45 | 0 | 50 | 155 | 485 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p05__sea_bg_0p20__sea_cfg_0p20 | 0.05 | 0.20 | 0.20 | 539.330 | 376.675 | 1.432 | 20.181 | 19.190 | 20.630 | 45 | 0 | 50 | 149 | 491 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p05__sea_bg_0p20__sea_cfg_0p20 | 0.05 | 0.20 | 0.20 | 542.662 | 377.301 | 1.438 | 24.476 | 21.880 | 26.460 | 45 | 0 | 50 | 155 | 485 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p05__sea_bg_0p20__sea_cfg_0p20 | 0.05 | 0.20 | 0.20 | 544.363 | 376.261 | 1.447 | 40.284 | 36.560 | 44.900 | 45 | 0 | 50 | 155 | 485 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p05__sea_bg_0p20__sea_cfg_0p20 | 0.05 | 0.20 | 0.20 | 544.006 | 376.503 | 1.445 | 23.739 | 23.250 | 24.350 | 45 | 0 | 50 | 155 | 485 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p05__sea_bg_0p20__sea_cfg_0p20 | 0.05 | 0.20 | 0.20 | 543.652 | 376.228 | 1.445 | 20.257 | 18.990 | 21.250 | 45 | 0 | 50 | 156 | 484 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p05__sea_bg_0p20__sea_cfg_0p20 | 0.05 | 0.20 | 0.20 | 544.915 | 375.127 | 1.453 | 24.568 | 23.510 | 25.060 | 45 | 0 | 50 | 157 | 483 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p05__sea_bg_0p20__sea_cfg_0p50 | 0.05 | 0.20 | 0.50 | 537.977 | 284.095 | 1.894 | 15.558 | 13.620 | 17.030 | 45 | 0 | 50 | 153 | 359 | 36 | 14 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p05__sea_bg_0p20__sea_cfg_0p50 | 0.05 | 0.20 | 0.50 | 538.450 | 284.928 | 1.890 | 20.014 | 17.930 | 22.950 | 45 | 0 | 50 | 153 | 359 | 36 | 14 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p05__sea_bg_0p20__sea_cfg_0p50 | 0.05 | 0.20 | 0.50 | 539.052 | 290.118 | 1.858 | 15.424 | 14.500 | 16.190 | 45 | 0 | 50 | 153 | 367 | 35 | 15 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p05__sea_bg_0p20__sea_cfg_0p50 | 0.05 | 0.20 | 0.50 | 538.562 | 284.358 | 1.894 | 22.348 | 21.050 | 22.930 | 45 | 0 | 50 | 153 | 359 | 36 | 14 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p05__sea_bg_0p20__sea_cfg_0p50 | 0.05 | 0.20 | 0.50 | 539.330 | 289.992 | 1.860 | 17.841 | 17.110 | 18.140 | 45 | 0 | 50 | 153 | 367 | 35 | 15 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p05__sea_bg_0p20__sea_cfg_0p50 | 0.05 | 0.20 | 0.50 | 542.662 | 293.171 | 1.851 | 17.240 | 15.550 | 19.590 | 45 | 0 | 50 | 153 | 367 | 35 | 15 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p05__sea_bg_0p20__sea_cfg_0p50 | 0.05 | 0.20 | 0.50 | 544.363 | 286.585 | 1.899 | 32.488 | 22.560 | 35.050 | 45 | 0 | 50 | 153 | 359 | 36 | 14 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p05__sea_bg_0p20__sea_cfg_0p50 | 0.05 | 0.20 | 0.50 | 544.006 | 287.192 | 1.894 | 21.321 | 20.550 | 21.790 | 45 | 0 | 50 | 153 | 359 | 36 | 14 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p05__sea_bg_0p20__sea_cfg_0p50 | 0.05 | 0.20 | 0.50 | 543.652 | 292.807 | 1.857 | 19.820 | 18.970 | 20.750 | 45 | 0 | 50 | 153 | 367 | 35 | 15 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p05__sea_bg_0p20__sea_cfg_0p50 | 0.05 | 0.20 | 0.50 | 544.915 | 286.278 | 1.903 | 25.011 | 23.820 | 25.540 | 45 | 0 | 50 | 154 | 358 | 36 | 14 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p05__sea_bg_0p50__sea_cfg_0p05 | 0.05 | 0.50 | 0.05 | 537.977 | 230.729 | 2.332 | 15.460 | 13.550 | 16.770 | 45 | 0 | 50 | 562 | 238 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p05__sea_bg_0p50__sea_cfg_0p05 | 0.05 | 0.50 | 0.05 | 538.450 | 234.561 | 2.296 | 18.945 | 17.210 | 20.990 | 45 | 0 | 50 | 556 | 244 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p05__sea_bg_0p50__sea_cfg_0p05 | 0.05 | 0.50 | 0.05 | 539.052 | 232.229 | 2.321 | 15.416 | 14.570 | 16.310 | 45 | 0 | 50 | 560 | 240 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p05__sea_bg_0p50__sea_cfg_0p05 | 0.05 | 0.50 | 0.05 | 538.562 | 230.684 | 2.335 | 22.032 | 20.690 | 22.550 | 45 | 0 | 50 | 562 | 238 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p05__sea_bg_0p50__sea_cfg_0p05 | 0.05 | 0.50 | 0.05 | 539.330 | 240.263 | 2.245 | 18.160 | 17.520 | 18.480 | 45 | 0 | 50 | 547 | 253 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p05__sea_bg_0p50__sea_cfg_0p05 | 0.05 | 0.50 | 0.05 | 542.662 | 241.941 | 2.243 | 17.006 | 15.500 | 19.150 | 45 | 0 | 50 | 548 | 252 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p05__sea_bg_0p50__sea_cfg_0p05 | 0.05 | 0.50 | 0.05 | 544.363 | 232.416 | 2.342 | 32.413 | 30.080 | 37.940 | 45 | 0 | 50 | 562 | 238 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p05__sea_bg_0p50__sea_cfg_0p05 | 0.05 | 0.50 | 0.05 | 544.006 | 232.615 | 2.339 | 21.142 | 20.410 | 21.750 | 45 | 0 | 50 | 562 | 238 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p05__sea_bg_0p50__sea_cfg_0p05 | 0.05 | 0.50 | 0.05 | 543.652 | 234.269 | 2.321 | 19.154 | 18.630 | 19.750 | 45 | 0 | 50 | 560 | 240 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p05__sea_bg_0p50__sea_cfg_0p05 | 0.05 | 0.50 | 0.05 | 544.915 | 241.926 | 2.252 | 24.804 | 23.480 | 25.360 | 45 | 0 | 50 | 548 | 252 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p05__sea_bg_0p50__sea_cfg_0p10 | 0.05 | 0.50 | 0.10 | 537.977 | 223.061 | 2.412 | 15.594 | 13.560 | 16.860 | 45 | 0 | 50 | 521 | 231 | 6 | 44 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p05__sea_bg_0p50__sea_cfg_0p10 | 0.05 | 0.50 | 0.10 | 538.450 | 231.166 | 2.329 | 18.906 | 17.110 | 20.870 | 45 | 0 | 50 | 516 | 244 | 5 | 45 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p05__sea_bg_0p50__sea_cfg_0p10 | 0.05 | 0.50 | 0.10 | 539.052 | 228.392 | 2.360 | 15.230 | 14.330 | 16.040 | 45 | 0 | 50 | 521 | 239 | 5 | 45 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p05__sea_bg_0p50__sea_cfg_0p10 | 0.05 | 0.50 | 0.10 | 538.562 | 222.761 | 2.418 | 22.003 | 20.830 | 22.400 | 45 | 0 | 50 | 521 | 231 | 6 | 44 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p05__sea_bg_0p50__sea_cfg_0p10 | 0.05 | 0.50 | 0.10 | 539.330 | 236.516 | 2.280 | 17.844 | 17.340 | 18.320 | 45 | 0 | 50 | 508 | 252 | 5 | 45 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p05__sea_bg_0p50__sea_cfg_0p10 | 0.05 | 0.50 | 0.10 | 542.662 | 238.298 | 2.277 | 16.902 | 15.160 | 18.880 | 45 | 0 | 50 | 509 | 251 | 5 | 45 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p05__sea_bg_0p50__sea_cfg_0p10 | 0.05 | 0.50 | 0.10 | 544.363 | 228.947 | 2.378 | 32.117 | 29.690 | 36.870 | 45 | 0 | 50 | 514 | 238 | 6 | 44 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p05__sea_bg_0p50__sea_cfg_0p10 | 0.05 | 0.50 | 0.10 | 544.006 | 229.514 | 2.370 | 20.824 | 20.200 | 21.380 | 45 | 0 | 50 | 522 | 238 | 5 | 45 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p05__sea_bg_0p50__sea_cfg_0p10 | 0.05 | 0.50 | 0.10 | 543.652 | 230.404 | 2.360 | 18.825 | 18.370 | 19.360 | 45 | 0 | 50 | 521 | 239 | 5 | 45 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p05__sea_bg_0p50__sea_cfg_0p10 | 0.05 | 0.50 | 0.10 | 544.915 | 237.841 | 2.291 | 23.910 | 22.480 | 24.400 | 45 | 0 | 50 | 500 | 252 | 6 | 44 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p05__sea_bg_0p50__sea_cfg_0p20 | 0.05 | 0.50 | 0.20 | 537.977 | 214.076 | 2.513 | 16.149 | 14.470 | 17.490 | 45 | 0 | 50 | 409 | 231 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p05__sea_bg_0p50__sea_cfg_0p20 | 0.05 | 0.50 | 0.20 | 538.450 | 217.860 | 2.472 | 18.974 | 17.200 | 20.700 | 45 | 0 | 50 | 403 | 237 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p05__sea_bg_0p50__sea_cfg_0p20 | 0.05 | 0.50 | 0.20 | 539.052 | 214.809 | 2.509 | 15.683 | 14.730 | 16.480 | 45 | 0 | 50 | 408 | 232 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p05__sea_bg_0p50__sea_cfg_0p20 | 0.05 | 0.50 | 0.20 | 538.562 | 214.054 | 2.516 | 20.922 | 20.300 | 21.390 | 45 | 0 | 50 | 409 | 231 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p05__sea_bg_0p50__sea_cfg_0p20 | 0.05 | 0.50 | 0.20 | 539.330 | 222.907 | 2.420 | 17.949 | 17.530 | 18.360 | 45 | 0 | 50 | 395 | 245 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p05__sea_bg_0p50__sea_cfg_0p20 | 0.05 | 0.50 | 0.20 | 542.662 | 223.774 | 2.425 | 17.171 | 15.430 | 19.480 | 45 | 0 | 50 | 397 | 243 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p05__sea_bg_0p50__sea_cfg_0p20 | 0.05 | 0.50 | 0.20 | 544.363 | 216.279 | 2.517 | 31.588 | 29.200 | 34.570 | 45 | 0 | 50 | 409 | 231 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p05__sea_bg_0p50__sea_cfg_0p20 | 0.05 | 0.50 | 0.20 | 544.006 | 215.920 | 2.519 | 20.481 | 19.950 | 21.170 | 45 | 0 | 50 | 409 | 231 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p05__sea_bg_0p50__sea_cfg_0p20 | 0.05 | 0.50 | 0.20 | 543.652 | 216.819 | 2.507 | 18.683 | 18.220 | 19.280 | 45 | 0 | 50 | 408 | 232 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p05__sea_bg_0p50__sea_cfg_0p20 | 0.05 | 0.50 | 0.20 | 544.915 | 224.400 | 2.428 | 23.282 | 22.190 | 23.820 | 45 | 0 | 50 | 396 | 244 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p05__sea_bg_0p50__sea_cfg_0p50 | 0.05 | 0.50 | 0.50 | 537.977 | 184.522 | 2.916 | 14.811 | 12.920 | 16.100 | 45 | 0 | 50 | 313 | 199 | 36 | 14 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p05__sea_bg_0p50__sea_cfg_0p50 | 0.05 | 0.50 | 0.50 | 538.450 | 189.348 | 2.844 | 17.418 | 16.110 | 18.930 | 45 | 0 | 50 | 314 | 206 | 35 | 15 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p05__sea_bg_0p50__sea_cfg_0p50 | 0.05 | 0.50 | 0.50 | 539.052 | 185.744 | 2.902 | 15.190 | 14.130 | 16.160 | 45 | 0 | 50 | 320 | 200 | 35 | 15 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p05__sea_bg_0p50__sea_cfg_0p50 | 0.05 | 0.50 | 0.50 | 538.562 | 184.295 | 2.922 | 20.597 | 19.090 | 21.060 | 45 | 0 | 50 | 313 | 199 | 36 | 14 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p05__sea_bg_0p50__sea_cfg_0p50 | 0.05 | 0.50 | 0.50 | 539.330 | 190.096 | 2.837 | 17.813 | 17.310 | 18.150 | 45 | 0 | 50 | 313 | 207 | 35 | 15 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p05__sea_bg_0p50__sea_cfg_0p50 | 0.05 | 0.50 | 0.50 | 542.662 | 191.141 | 2.839 | 16.321 | 14.650 | 19.130 | 45 | 0 | 50 | 314 | 206 | 35 | 15 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p05__sea_bg_0p50__sea_cfg_0p50 | 0.05 | 0.50 | 0.50 | 544.363 | 189.693 | 2.870 | 28.192 | 21.710 | 30.510 | 45 | 0 | 50 | 307 | 205 | 36 | 14 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p05__sea_bg_0p50__sea_cfg_0p50 | 0.05 | 0.50 | 0.50 | 544.006 | 185.978 | 2.925 | 18.910 | 17.840 | 19.480 | 45 | 0 | 50 | 313 | 199 | 36 | 14 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p05__sea_bg_0p50__sea_cfg_0p50 | 0.05 | 0.50 | 0.50 | 543.652 | 187.424 | 2.901 | 17.781 | 16.980 | 18.500 | 45 | 0 | 50 | 320 | 200 | 35 | 15 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p05__sea_bg_0p50__sea_cfg_0p50 | 0.05 | 0.50 | 0.50 | 544.915 | 190.322 | 2.863 | 23.681 | 22.800 | 24.270 | 45 | 0 | 50 | 306 | 206 | 36 | 14 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p10__sea_bg_0p05__sea_cfg_0p05 | 0.10 | 0.05 | 0.05 | 537.977 | 519.694 | 1.035 | 32.608 | 28.460 | 36.510 | 45 | 6 | 44 | 0 | 704 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p10__sea_bg_0p05__sea_cfg_0p05 | 0.10 | 0.05 | 0.05 | 538.450 | 519.823 | 1.036 | 42.542 | 40.060 | 43.460 | 45 | 6 | 44 | 0 | 704 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p10__sea_bg_0p05__sea_cfg_0p05 | 0.10 | 0.05 | 0.05 | 539.052 | 530.549 | 1.016 | 31.720 | 27.970 | 33.960 | 45 | 5 | 45 | 0 | 720 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p10__sea_bg_0p05__sea_cfg_0p05 | 0.10 | 0.05 | 0.05 | 538.562 | 519.635 | 1.036 | 36.091 | 33.440 | 37.350 | 45 | 6 | 44 | 0 | 704 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p10__sea_bg_0p05__sea_cfg_0p05 | 0.10 | 0.05 | 0.05 | 539.330 | 530.200 | 1.017 | 30.030 | 28.400 | 30.990 | 45 | 5 | 45 | 0 | 720 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p10__sea_bg_0p05__sea_cfg_0p05 | 0.10 | 0.05 | 0.05 | 542.662 | 535.214 | 1.014 | 35.110 | 33.140 | 38.620 | 45 | 5 | 45 | 0 | 720 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p10__sea_bg_0p05__sea_cfg_0p05 | 0.10 | 0.05 | 0.05 | 544.363 | 524.144 | 1.039 | 46.819 | 42.980 | 49.490 | 45 | 6 | 44 | 0 | 704 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p10__sea_bg_0p05__sea_cfg_0p05 | 0.10 | 0.05 | 0.05 | 544.006 | 524.433 | 1.037 | 31.390 | 30.470 | 32.180 | 45 | 6 | 44 | 0 | 704 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p10__sea_bg_0p05__sea_cfg_0p05 | 0.10 | 0.05 | 0.05 | 543.652 | 535.115 | 1.016 | 32.880 | 31.630 | 33.980 | 45 | 5 | 45 | 0 | 720 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p10__sea_bg_0p05__sea_cfg_0p05 | 0.10 | 0.05 | 0.05 | 544.915 | 524.527 | 1.039 | 39.783 | 38.470 | 40.700 | 45 | 6 | 44 | 0 | 704 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p10__sea_bg_0p05__sea_cfg_0p10 | 0.10 | 0.05 | 0.10 | 537.977 | 517.282 | 1.040 | 33.457 | 28.510 | 37.320 | 45 | 6 | 44 | 0 | 704 | 6 | 44 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p10__sea_bg_0p05__sea_cfg_0p10 | 0.10 | 0.05 | 0.10 | 538.450 | 517.241 | 1.041 | 42.599 | 39.970 | 43.550 | 45 | 6 | 44 | 0 | 704 | 6 | 44 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p10__sea_bg_0p05__sea_cfg_0p10 | 0.10 | 0.05 | 0.10 | 539.052 | 528.903 | 1.019 | 31.875 | 28.940 | 33.960 | 45 | 5 | 45 | 0 | 720 | 5 | 45 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p10__sea_bg_0p05__sea_cfg_0p10 | 0.10 | 0.05 | 0.10 | 538.562 | 517.545 | 1.041 | 36.401 | 33.530 | 37.600 | 45 | 6 | 44 | 0 | 704 | 6 | 44 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p10__sea_bg_0p05__sea_cfg_0p10 | 0.10 | 0.05 | 0.10 | 539.330 | 528.400 | 1.021 | 30.520 | 28.450 | 31.390 | 45 | 5 | 45 | 0 | 720 | 5 | 45 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p10__sea_bg_0p05__sea_cfg_0p10 | 0.10 | 0.05 | 0.10 | 542.662 | 533.053 | 1.018 | 35.343 | 33.280 | 38.960 | 45 | 5 | 45 | 0 | 720 | 5 | 45 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p10__sea_bg_0p05__sea_cfg_0p10 | 0.10 | 0.05 | 0.10 | 544.363 | 522.272 | 1.042 | 47.179 | 44.800 | 49.260 | 45 | 6 | 44 | 0 | 704 | 6 | 44 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p10__sea_bg_0p05__sea_cfg_0p10 | 0.10 | 0.05 | 0.10 | 544.006 | 522.161 | 1.042 | 31.285 | 30.160 | 32.150 | 45 | 6 | 44 | 0 | 704 | 6 | 44 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p10__sea_bg_0p05__sea_cfg_0p10 | 0.10 | 0.05 | 0.10 | 543.652 | 533.901 | 1.018 | 32.634 | 31.370 | 33.790 | 45 | 5 | 45 | 0 | 720 | 5 | 45 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p10__sea_bg_0p05__sea_cfg_0p10 | 0.10 | 0.05 | 0.10 | 544.915 | 522.424 | 1.043 | 39.833 | 38.530 | 40.700 | 45 | 6 | 44 | 0 | 704 | 6 | 44 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p10__sea_bg_0p05__sea_cfg_0p20 | 0.10 | 0.05 | 0.20 | 537.977 | 439.035 | 1.225 | 19.126 | 16.470 | 22.090 | 45 | 6 | 46 | 0 | 592 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p10__sea_bg_0p05__sea_cfg_0p20 | 0.10 | 0.05 | 0.20 | 538.450 | 439.085 | 1.226 | 32.161 | 25.440 | 36.200 | 45 | 6 | 46 | 0 | 592 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p10__sea_bg_0p05__sea_cfg_0p20 | 0.10 | 0.05 | 0.20 | 539.052 | 444.614 | 1.212 | 21.796 | 19.440 | 24.740 | 45 | 5 | 46 | 0 | 600 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p10__sea_bg_0p05__sea_cfg_0p20 | 0.10 | 0.05 | 0.20 | 538.562 | 439.159 | 1.226 | 25.678 | 24.880 | 26.090 | 45 | 6 | 46 | 0 | 592 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p10__sea_bg_0p05__sea_cfg_0p20 | 0.10 | 0.05 | 0.20 | 539.330 | 444.204 | 1.214 | 22.461 | 21.000 | 23.020 | 45 | 5 | 46 | 0 | 600 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p10__sea_bg_0p05__sea_cfg_0p20 | 0.10 | 0.05 | 0.20 | 542.662 | 448.295 | 1.211 | 25.712 | 23.990 | 28.590 | 45 | 5 | 46 | 0 | 600 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p10__sea_bg_0p05__sea_cfg_0p20 | 0.10 | 0.05 | 0.20 | 544.363 | 442.647 | 1.230 | 41.539 | 37.450 | 46.390 | 45 | 6 | 46 | 0 | 592 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p10__sea_bg_0p05__sea_cfg_0p20 | 0.10 | 0.05 | 0.20 | 544.006 | 443.018 | 1.228 | 24.973 | 23.810 | 25.730 | 45 | 6 | 46 | 0 | 592 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p10__sea_bg_0p05__sea_cfg_0p20 | 0.10 | 0.05 | 0.20 | 543.652 | 448.797 | 1.211 | 20.312 | 19.100 | 21.070 | 45 | 5 | 46 | 0 | 600 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p10__sea_bg_0p05__sea_cfg_0p20 | 0.10 | 0.05 | 0.20 | 544.915 | 443.518 | 1.229 | 30.111 | 27.180 | 31.450 | 45 | 6 | 46 | 0 | 592 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p10__sea_bg_0p05__sea_cfg_0p50 | 0.10 | 0.05 | 0.50 | 537.977 | 349.085 | 1.541 | 14.872 | 13.080 | 16.340 | 45 | 6 | 45 | 0 | 464 | 36 | 14 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p10__sea_bg_0p05__sea_cfg_0p50 | 0.10 | 0.05 | 0.50 | 538.450 | 348.912 | 1.543 | 20.262 | 18.190 | 23.090 | 45 | 6 | 45 | 0 | 464 | 36 | 14 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p10__sea_bg_0p05__sea_cfg_0p50 | 0.10 | 0.05 | 0.50 | 539.052 | 355.190 | 1.518 | 15.113 | 14.180 | 15.920 | 45 | 6 | 44 | 0 | 472 | 35 | 15 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p10__sea_bg_0p05__sea_cfg_0p50 | 0.10 | 0.05 | 0.50 | 538.562 | 349.187 | 1.542 | 24.344 | 23.210 | 25.150 | 45 | 6 | 45 | 0 | 464 | 36 | 14 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p10__sea_bg_0p05__sea_cfg_0p50 | 0.10 | 0.05 | 0.50 | 539.330 | 360.222 | 1.497 | 18.153 | 17.350 | 18.610 | 45 | 5 | 45 | 0 | 480 | 35 | 15 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p10__sea_bg_0p05__sea_cfg_0p50 | 0.10 | 0.05 | 0.50 | 542.662 | 363.036 | 1.495 | 16.875 | 14.910 | 19.360 | 45 | 5 | 45 | 0 | 480 | 35 | 15 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p10__sea_bg_0p05__sea_cfg_0p50 | 0.10 | 0.05 | 0.50 | 544.363 | 352.053 | 1.546 | 32.644 | 22.830 | 38.060 | 45 | 6 | 45 | 0 | 464 | 36 | 14 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p10__sea_bg_0p05__sea_cfg_0p50 | 0.10 | 0.05 | 0.50 | 544.006 | 352.293 | 1.544 | 20.884 | 20.100 | 21.580 | 45 | 6 | 45 | 0 | 464 | 36 | 14 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p10__sea_bg_0p05__sea_cfg_0p50 | 0.10 | 0.05 | 0.50 | 543.652 | 363.589 | 1.495 | 19.219 | 18.580 | 19.820 | 45 | 5 | 45 | 0 | 480 | 35 | 15 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p10__sea_bg_0p05__sea_cfg_0p50 | 0.10 | 0.05 | 0.50 | 544.915 | 352.026 | 1.548 | 25.340 | 24.150 | 26.090 | 45 | 6 | 45 | 0 | 464 | 36 | 14 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p10__sea_bg_0p10__sea_cfg_0p05 | 0.10 | 0.10 | 0.05 | 537.977 | 520.024 | 1.035 | 32.608 | 28.460 | 36.510 | 45 | 6 | 44 | 0 | 704 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p10__sea_bg_0p10__sea_cfg_0p05 | 0.10 | 0.10 | 0.05 | 538.450 | 520.089 | 1.035 | 42.542 | 40.060 | 43.460 | 45 | 6 | 44 | 0 | 704 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p10__sea_bg_0p10__sea_cfg_0p05 | 0.10 | 0.10 | 0.05 | 539.052 | 531.002 | 1.015 | 31.720 | 27.970 | 33.960 | 45 | 5 | 45 | 0 | 720 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p10__sea_bg_0p10__sea_cfg_0p05 | 0.10 | 0.10 | 0.05 | 538.562 | 520.389 | 1.035 | 36.091 | 33.440 | 37.350 | 45 | 6 | 44 | 0 | 704 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p10__sea_bg_0p10__sea_cfg_0p05 | 0.10 | 0.10 | 0.05 | 539.330 | 530.814 | 1.016 | 30.030 | 28.400 | 30.990 | 45 | 5 | 45 | 0 | 720 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p10__sea_bg_0p10__sea_cfg_0p05 | 0.10 | 0.10 | 0.05 | 542.662 | 535.091 | 1.014 | 35.110 | 33.140 | 38.620 | 45 | 5 | 45 | 0 | 720 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p10__sea_bg_0p10__sea_cfg_0p05 | 0.10 | 0.10 | 0.05 | 544.363 | 525.020 | 1.037 | 46.819 | 42.980 | 49.490 | 45 | 6 | 44 | 0 | 704 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p10__sea_bg_0p10__sea_cfg_0p05 | 0.10 | 0.10 | 0.05 | 544.006 | 524.903 | 1.036 | 31.390 | 30.470 | 32.180 | 45 | 6 | 44 | 0 | 704 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p10__sea_bg_0p10__sea_cfg_0p05 | 0.10 | 0.10 | 0.05 | 543.652 | 536.020 | 1.014 | 32.880 | 31.630 | 33.980 | 45 | 5 | 45 | 0 | 720 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p10__sea_bg_0p10__sea_cfg_0p05 | 0.10 | 0.10 | 0.05 | 544.915 | 524.698 | 1.039 | 39.783 | 38.470 | 40.700 | 45 | 6 | 44 | 0 | 704 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p10__sea_bg_0p10__sea_cfg_0p10 | 0.10 | 0.10 | 0.10 | 537.977 | 516.967 | 1.041 | 33.457 | 28.510 | 37.320 | 45 | 6 | 44 | 0 | 704 | 6 | 44 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p10__sea_bg_0p10__sea_cfg_0p10 | 0.10 | 0.10 | 0.10 | 538.450 | 517.746 | 1.040 | 42.599 | 39.970 | 43.550 | 45 | 6 | 44 | 0 | 704 | 6 | 44 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p10__sea_bg_0p10__sea_cfg_0p10 | 0.10 | 0.10 | 0.10 | 539.052 | 528.913 | 1.019 | 31.875 | 28.940 | 33.960 | 45 | 5 | 45 | 0 | 720 | 5 | 45 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p10__sea_bg_0p10__sea_cfg_0p10 | 0.10 | 0.10 | 0.10 | 538.562 | 517.920 | 1.040 | 36.401 | 33.530 | 37.600 | 45 | 6 | 44 | 0 | 704 | 6 | 44 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p10__sea_bg_0p10__sea_cfg_0p10 | 0.10 | 0.10 | 0.10 | 539.330 | 528.750 | 1.020 | 30.520 | 28.450 | 31.390 | 45 | 5 | 45 | 0 | 720 | 5 | 45 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p10__sea_bg_0p10__sea_cfg_0p10 | 0.10 | 0.10 | 0.10 | 542.662 | 533.404 | 1.017 | 35.343 | 33.280 | 38.960 | 45 | 5 | 45 | 0 | 720 | 5 | 45 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p10__sea_bg_0p10__sea_cfg_0p10 | 0.10 | 0.10 | 0.10 | 544.363 | 522.864 | 1.041 | 47.179 | 44.800 | 49.260 | 45 | 6 | 44 | 0 | 704 | 6 | 44 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p10__sea_bg_0p10__sea_cfg_0p10 | 0.10 | 0.10 | 0.10 | 544.006 | 522.265 | 1.042 | 31.285 | 30.160 | 32.150 | 45 | 6 | 44 | 0 | 704 | 6 | 44 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p10__sea_bg_0p10__sea_cfg_0p10 | 0.10 | 0.10 | 0.10 | 543.652 | 533.769 | 1.019 | 32.634 | 31.370 | 33.790 | 45 | 5 | 45 | 0 | 720 | 5 | 45 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p10__sea_bg_0p10__sea_cfg_0p10 | 0.10 | 0.10 | 0.10 | 544.915 | 522.213 | 1.043 | 39.833 | 38.530 | 40.700 | 45 | 6 | 44 | 0 | 704 | 6 | 44 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p10__sea_bg_0p10__sea_cfg_0p20 | 0.10 | 0.10 | 0.20 | 537.977 | 438.775 | 1.226 | 19.126 | 16.470 | 22.090 | 45 | 6 | 46 | 0 | 592 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p10__sea_bg_0p10__sea_cfg_0p20 | 0.10 | 0.10 | 0.20 | 538.450 | 439.238 | 1.226 | 32.161 | 25.440 | 36.200 | 45 | 6 | 46 | 0 | 592 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p10__sea_bg_0p10__sea_cfg_0p20 | 0.10 | 0.10 | 0.20 | 539.052 | 444.826 | 1.212 | 21.796 | 19.440 | 24.740 | 45 | 5 | 46 | 0 | 600 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p10__sea_bg_0p10__sea_cfg_0p20 | 0.10 | 0.10 | 0.20 | 538.562 | 439.453 | 1.226 | 25.678 | 24.880 | 26.090 | 45 | 6 | 46 | 0 | 592 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p10__sea_bg_0p10__sea_cfg_0p20 | 0.10 | 0.10 | 0.20 | 539.330 | 444.760 | 1.213 | 22.461 | 21.000 | 23.020 | 45 | 5 | 46 | 0 | 600 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p10__sea_bg_0p10__sea_cfg_0p20 | 0.10 | 0.10 | 0.20 | 542.662 | 448.541 | 1.210 | 25.712 | 23.990 | 28.590 | 45 | 5 | 46 | 0 | 600 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p10__sea_bg_0p10__sea_cfg_0p20 | 0.10 | 0.10 | 0.20 | 544.363 | 443.177 | 1.228 | 41.539 | 37.450 | 46.390 | 45 | 6 | 46 | 0 | 592 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p10__sea_bg_0p10__sea_cfg_0p20 | 0.10 | 0.10 | 0.20 | 544.006 | 442.980 | 1.228 | 24.973 | 23.810 | 25.730 | 45 | 6 | 46 | 0 | 592 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p10__sea_bg_0p10__sea_cfg_0p20 | 0.10 | 0.10 | 0.20 | 543.652 | 448.797 | 1.211 | 20.312 | 19.100 | 21.070 | 45 | 5 | 46 | 0 | 600 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p10__sea_bg_0p10__sea_cfg_0p20 | 0.10 | 0.10 | 0.20 | 544.915 | 443.096 | 1.230 | 30.111 | 27.180 | 31.450 | 45 | 6 | 46 | 0 | 592 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p10__sea_bg_0p10__sea_cfg_0p50 | 0.10 | 0.10 | 0.50 | 537.977 | 348.878 | 1.542 | 14.872 | 13.080 | 16.340 | 45 | 6 | 45 | 0 | 464 | 36 | 14 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p10__sea_bg_0p10__sea_cfg_0p50 | 0.10 | 0.10 | 0.50 | 538.450 | 349.120 | 1.542 | 20.262 | 18.190 | 23.090 | 45 | 6 | 45 | 0 | 464 | 36 | 14 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p10__sea_bg_0p10__sea_cfg_0p50 | 0.10 | 0.10 | 0.50 | 539.052 | 354.665 | 1.520 | 15.113 | 14.180 | 15.920 | 45 | 6 | 44 | 0 | 472 | 35 | 15 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p10__sea_bg_0p10__sea_cfg_0p50 | 0.10 | 0.10 | 0.50 | 538.562 | 349.598 | 1.541 | 24.344 | 23.210 | 25.150 | 45 | 6 | 45 | 0 | 464 | 36 | 14 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p10__sea_bg_0p10__sea_cfg_0p50 | 0.10 | 0.10 | 0.50 | 539.330 | 360.461 | 1.496 | 18.153 | 17.350 | 18.610 | 45 | 5 | 45 | 0 | 480 | 35 | 15 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p10__sea_bg_0p10__sea_cfg_0p50 | 0.10 | 0.10 | 0.50 | 542.662 | 363.230 | 1.494 | 16.875 | 14.910 | 19.360 | 45 | 5 | 45 | 0 | 480 | 35 | 15 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p10__sea_bg_0p10__sea_cfg_0p50 | 0.10 | 0.10 | 0.50 | 544.363 | 352.467 | 1.544 | 32.644 | 22.830 | 38.060 | 45 | 6 | 45 | 0 | 464 | 36 | 14 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p10__sea_bg_0p10__sea_cfg_0p50 | 0.10 | 0.10 | 0.50 | 544.006 | 352.332 | 1.544 | 20.884 | 20.100 | 21.580 | 45 | 6 | 45 | 0 | 464 | 36 | 14 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p10__sea_bg_0p10__sea_cfg_0p50 | 0.10 | 0.10 | 0.50 | 543.652 | 362.820 | 1.498 | 19.219 | 18.580 | 19.820 | 45 | 5 | 45 | 0 | 480 | 35 | 15 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p10__sea_bg_0p10__sea_cfg_0p50 | 0.10 | 0.10 | 0.50 | 544.915 | 352.329 | 1.547 | 25.340 | 24.150 | 26.090 | 45 | 6 | 45 | 0 | 464 | 36 | 14 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p10__sea_bg_0p20__sea_cfg_0p05 | 0.10 | 0.20 | 0.05 | 537.977 | 369.905 | 1.454 | 20.526 | 17.710 | 23.360 | 45 | 6 | 44 | 240 | 464 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p10__sea_bg_0p20__sea_cfg_0p05 | 0.10 | 0.20 | 0.05 | 538.450 | 370.311 | 1.454 | 31.673 | 25.390 | 36.120 | 45 | 6 | 44 | 240 | 464 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p10__sea_bg_0p20__sea_cfg_0p05 | 0.10 | 0.20 | 0.05 | 539.052 | 370.217 | 1.456 | 20.276 | 17.510 | 23.420 | 45 | 5 | 45 | 256 | 464 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p10__sea_bg_0p20__sea_cfg_0p05 | 0.10 | 0.20 | 0.05 | 538.562 | 370.459 | 1.454 | 25.052 | 24.250 | 25.700 | 45 | 6 | 44 | 240 | 464 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p10__sea_bg_0p20__sea_cfg_0p05 | 0.10 | 0.20 | 0.05 | 539.330 | 375.026 | 1.438 | 21.200 | 19.980 | 21.600 | 45 | 5 | 45 | 250 | 470 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p10__sea_bg_0p20__sea_cfg_0p05 | 0.10 | 0.20 | 0.05 | 542.662 | 373.099 | 1.454 | 24.599 | 21.960 | 26.370 | 45 | 5 | 45 | 256 | 464 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p10__sea_bg_0p20__sea_cfg_0p05 | 0.10 | 0.20 | 0.05 | 544.363 | 382.356 | 1.424 | 41.260 | 37.600 | 45.720 | 45 | 6 | 44 | 226 | 478 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p10__sea_bg_0p20__sea_cfg_0p05 | 0.10 | 0.20 | 0.05 | 544.006 | 373.416 | 1.457 | 24.400 | 23.840 | 25.150 | 45 | 6 | 44 | 240 | 464 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p10__sea_bg_0p20__sea_cfg_0p05 | 0.10 | 0.20 | 0.05 | 543.652 | 373.459 | 1.456 | 20.542 | 19.180 | 21.520 | 45 | 5 | 45 | 256 | 464 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p10__sea_bg_0p20__sea_cfg_0p05 | 0.10 | 0.20 | 0.05 | 544.915 | 376.827 | 1.446 | 26.384 | 25.330 | 27.000 | 45 | 6 | 44 | 235 | 469 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p10__sea_bg_0p20__sea_cfg_0p10 | 0.10 | 0.20 | 0.10 | 537.977 | 367.548 | 1.464 | 20.533 | 17.760 | 23.310 | 45 | 6 | 44 | 240 | 464 | 6 | 44 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p10__sea_bg_0p20__sea_cfg_0p10 | 0.10 | 0.20 | 0.10 | 538.450 | 367.742 | 1.464 | 31.658 | 25.400 | 36.110 | 45 | 6 | 44 | 240 | 464 | 6 | 44 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p10__sea_bg_0p20__sea_cfg_0p10 | 0.10 | 0.20 | 0.10 | 539.052 | 367.881 | 1.465 | 20.272 | 17.500 | 23.420 | 45 | 5 | 45 | 256 | 464 | 5 | 45 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p10__sea_bg_0p20__sea_cfg_0p10 | 0.10 | 0.20 | 0.10 | 538.562 | 367.517 | 1.465 | 25.043 | 24.210 | 25.750 | 45 | 6 | 44 | 240 | 464 | 6 | 44 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p10__sea_bg_0p20__sea_cfg_0p10 | 0.10 | 0.20 | 0.10 | 539.330 | 372.404 | 1.448 | 21.204 | 20.000 | 21.590 | 45 | 5 | 45 | 250 | 470 | 5 | 45 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p10__sea_bg_0p20__sea_cfg_0p10 | 0.10 | 0.20 | 0.10 | 542.662 | 371.119 | 1.462 | 24.592 | 22.050 | 26.330 | 45 | 5 | 45 | 256 | 464 | 5 | 45 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p10__sea_bg_0p20__sea_cfg_0p10 | 0.10 | 0.20 | 0.10 | 544.363 | 379.421 | 1.435 | 41.222 | 37.520 | 45.580 | 45 | 6 | 44 | 226 | 478 | 6 | 44 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p10__sea_bg_0p20__sea_cfg_0p10 | 0.10 | 0.20 | 0.10 | 544.006 | 370.498 | 1.468 | 24.383 | 23.790 | 25.170 | 45 | 6 | 44 | 240 | 464 | 6 | 44 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p10__sea_bg_0p20__sea_cfg_0p10 | 0.10 | 0.20 | 0.10 | 543.652 | 371.358 | 1.464 | 20.545 | 19.170 | 21.540 | 45 | 5 | 45 | 256 | 464 | 5 | 45 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p10__sea_bg_0p20__sea_cfg_0p10 | 0.10 | 0.20 | 0.10 | 544.915 | 374.174 | 1.456 | 26.402 | 25.310 | 27.040 | 45 | 6 | 44 | 235 | 469 | 6 | 44 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p10__sea_bg_0p20__sea_cfg_0p20 | 0.10 | 0.20 | 0.20 | 537.977 | 362.237 | 1.485 | 20.120 | 17.440 | 23.090 | 45 | 6 | 46 | 123 | 469 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p10__sea_bg_0p20__sea_cfg_0p20 | 0.10 | 0.20 | 0.20 | 538.450 | 362.591 | 1.485 | 29.440 | 26.660 | 31.730 | 45 | 6 | 46 | 122 | 470 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p10__sea_bg_0p20__sea_cfg_0p20 | 0.10 | 0.20 | 0.20 | 539.052 | 362.221 | 1.488 | 20.223 | 17.670 | 23.080 | 45 | 6 | 46 | 122 | 470 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p10__sea_bg_0p20__sea_cfg_0p20 | 0.10 | 0.20 | 0.20 | 538.562 | 362.992 | 1.484 | 24.496 | 23.650 | 24.850 | 45 | 6 | 46 | 122 | 470 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p10__sea_bg_0p20__sea_cfg_0p20 | 0.10 | 0.20 | 0.20 | 539.330 | 366.022 | 1.473 | 20.245 | 19.420 | 20.680 | 45 | 6 | 46 | 116 | 476 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p10__sea_bg_0p20__sea_cfg_0p20 | 0.10 | 0.20 | 0.20 | 542.662 | 365.815 | 1.483 | 24.353 | 21.930 | 25.950 | 45 | 5 | 46 | 130 | 470 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p10__sea_bg_0p20__sea_cfg_0p20 | 0.10 | 0.20 | 0.20 | 544.363 | 370.560 | 1.469 | 39.885 | 35.770 | 44.290 | 45 | 6 | 46 | 115 | 477 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p10__sea_bg_0p20__sea_cfg_0p20 | 0.10 | 0.20 | 0.20 | 544.006 | 365.734 | 1.487 | 23.314 | 22.760 | 23.700 | 45 | 6 | 46 | 122 | 470 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p10__sea_bg_0p20__sea_cfg_0p20 | 0.10 | 0.20 | 0.20 | 543.652 | 364.991 | 1.489 | 20.272 | 19.010 | 21.270 | 45 | 6 | 46 | 123 | 469 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p10__sea_bg_0p20__sea_cfg_0p20 | 0.10 | 0.20 | 0.20 | 544.915 | 364.958 | 1.493 | 24.630 | 23.760 | 25.130 | 45 | 6 | 46 | 124 | 468 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p10__sea_bg_0p20__sea_cfg_0p50 | 0.10 | 0.20 | 0.50 | 537.977 | 273.754 | 1.965 | 15.573 | 13.680 | 16.940 | 45 | 6 | 45 | 120 | 344 | 36 | 14 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p10__sea_bg_0p20__sea_cfg_0p50 | 0.10 | 0.20 | 0.50 | 538.450 | 273.736 | 1.967 | 19.968 | 17.860 | 22.840 | 45 | 6 | 45 | 120 | 344 | 36 | 14 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p10__sea_bg_0p20__sea_cfg_0p50 | 0.10 | 0.20 | 0.50 | 539.052 | 279.413 | 1.929 | 15.669 | 14.710 | 16.490 | 45 | 6 | 44 | 120 | 352 | 35 | 15 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p10__sea_bg_0p20__sea_cfg_0p50 | 0.10 | 0.20 | 0.50 | 538.562 | 274.388 | 1.963 | 22.137 | 21.040 | 22.670 | 45 | 6 | 45 | 120 | 344 | 36 | 14 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p10__sea_bg_0p20__sea_cfg_0p50 | 0.10 | 0.20 | 0.50 | 539.330 | 279.433 | 1.930 | 18.045 | 17.330 | 18.330 | 45 | 6 | 44 | 120 | 352 | 35 | 15 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p10__sea_bg_0p20__sea_cfg_0p50 | 0.10 | 0.20 | 0.50 | 542.662 | 282.022 | 1.924 | 17.314 | 15.670 | 19.500 | 45 | 5 | 45 | 128 | 352 | 35 | 15 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p10__sea_bg_0p20__sea_cfg_0p50 | 0.10 | 0.20 | 0.50 | 544.363 | 281.147 | 1.936 | 32.381 | 22.510 | 34.900 | 45 | 6 | 45 | 113 | 351 | 36 | 14 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p10__sea_bg_0p20__sea_cfg_0p50 | 0.10 | 0.20 | 0.50 | 544.006 | 276.522 | 1.967 | 21.441 | 20.570 | 21.990 | 45 | 6 | 45 | 120 | 344 | 36 | 14 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p10__sea_bg_0p20__sea_cfg_0p50 | 0.10 | 0.20 | 0.50 | 543.652 | 282.058 | 1.927 | 19.945 | 19.090 | 20.850 | 45 | 6 | 44 | 120 | 352 | 35 | 15 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p10__sea_bg_0p20__sea_cfg_0p50 | 0.10 | 0.20 | 0.50 | 544.915 | 275.953 | 1.975 | 24.900 | 24.030 | 25.440 | 45 | 6 | 45 | 121 | 343 | 36 | 14 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p10__sea_bg_0p50__sea_cfg_0p05 | 0.10 | 0.50 | 0.05 | 537.977 | 219.433 | 2.452 | 15.429 | 13.550 | 16.770 | 45 | 6 | 44 | 480 | 224 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p10__sea_bg_0p50__sea_cfg_0p05 | 0.10 | 0.50 | 0.05 | 538.450 | 232.394 | 2.317 | 18.983 | 17.240 | 21.050 | 45 | 6 | 44 | 460 | 244 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p10__sea_bg_0p50__sea_cfg_0p05 | 0.10 | 0.50 | 0.05 | 539.052 | 228.624 | 2.358 | 15.530 | 14.690 | 16.420 | 45 | 6 | 44 | 466 | 238 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p10__sea_bg_0p50__sea_cfg_0p05 | 0.10 | 0.50 | 0.05 | 538.562 | 219.479 | 2.454 | 21.953 | 20.620 | 22.470 | 45 | 6 | 44 | 480 | 224 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p10__sea_bg_0p50__sea_cfg_0p05 | 0.10 | 0.50 | 0.05 | 539.330 | 236.660 | 2.279 | 18.163 | 17.640 | 18.560 | 45 | 6 | 44 | 453 | 251 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p10__sea_bg_0p50__sea_cfg_0p05 | 0.10 | 0.50 | 0.05 | 542.662 | 237.974 | 2.280 | 17.041 | 15.580 | 19.100 | 45 | 5 | 45 | 470 | 250 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p10__sea_bg_0p50__sea_cfg_0p05 | 0.10 | 0.50 | 0.05 | 544.363 | 230.432 | 2.362 | 32.703 | 30.050 | 37.760 | 45 | 6 | 44 | 466 | 238 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p10__sea_bg_0p50__sea_cfg_0p05 | 0.10 | 0.50 | 0.05 | 544.006 | 230.498 | 2.360 | 21.159 | 20.400 | 21.740 | 45 | 6 | 44 | 466 | 238 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p10__sea_bg_0p50__sea_cfg_0p05 | 0.10 | 0.50 | 0.05 | 543.652 | 230.346 | 2.360 | 19.219 | 18.700 | 19.760 | 45 | 6 | 44 | 466 | 238 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p10__sea_bg_0p50__sea_cfg_0p05 | 0.10 | 0.50 | 0.05 | 544.915 | 239.493 | 2.275 | 24.708 | 23.410 | 25.340 | 45 | 6 | 44 | 452 | 252 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p10__sea_bg_0p50__sea_cfg_0p10 | 0.10 | 0.50 | 0.10 | 537.977 | 216.922 | 2.480 | 15.429 | 13.560 | 16.770 | 45 | 6 | 44 | 480 | 224 | 6 | 44 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p10__sea_bg_0p50__sea_cfg_0p10 | 0.10 | 0.50 | 0.10 | 538.450 | 229.533 | 2.346 | 18.982 | 17.240 | 21.050 | 45 | 6 | 44 | 460 | 244 | 6 | 44 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p10__sea_bg_0p50__sea_cfg_0p10 | 0.10 | 0.50 | 0.10 | 539.052 | 226.006 | 2.385 | 15.527 | 14.690 | 16.410 | 45 | 6 | 44 | 466 | 238 | 6 | 44 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p10__sea_bg_0p50__sea_cfg_0p10 | 0.10 | 0.50 | 0.10 | 538.562 | 217.134 | 2.480 | 21.953 | 20.630 | 22.500 | 45 | 6 | 44 | 480 | 224 | 6 | 44 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p10__sea_bg_0p50__sea_cfg_0p10 | 0.10 | 0.50 | 0.10 | 539.330 | 234.133 | 2.304 | 18.171 | 17.650 | 18.560 | 45 | 6 | 44 | 453 | 251 | 6 | 44 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p10__sea_bg_0p50__sea_cfg_0p10 | 0.10 | 0.50 | 0.10 | 542.662 | 236.043 | 2.299 | 17.041 | 15.600 | 19.080 | 45 | 5 | 45 | 470 | 250 | 5 | 45 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p10__sea_bg_0p50__sea_cfg_0p10 | 0.10 | 0.50 | 0.10 | 544.363 | 227.828 | 2.389 | 32.711 | 30.050 | 37.760 | 45 | 6 | 44 | 466 | 238 | 6 | 44 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p10__sea_bg_0p50__sea_cfg_0p10 | 0.10 | 0.50 | 0.10 | 544.006 | 227.955 | 2.386 | 21.176 | 20.420 | 21.770 | 45 | 6 | 44 | 466 | 238 | 6 | 44 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p10__sea_bg_0p50__sea_cfg_0p10 | 0.10 | 0.50 | 0.10 | 543.652 | 227.817 | 2.386 | 19.218 | 18.700 | 19.750 | 45 | 6 | 44 | 466 | 238 | 6 | 44 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p10__sea_bg_0p50__sea_cfg_0p10 | 0.10 | 0.50 | 0.10 | 544.915 | 236.880 | 2.300 | 24.708 | 23.410 | 25.350 | 45 | 6 | 44 | 452 | 252 | 6 | 44 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p10__sea_bg_0p50__sea_cfg_0p20 | 0.10 | 0.50 | 0.20 | 537.977 | 208.344 | 2.582 | 16.214 | 14.670 | 17.620 | 45 | 6 | 46 | 368 | 224 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p10__sea_bg_0p50__sea_cfg_0p20 | 0.10 | 0.50 | 0.20 | 538.450 | 216.496 | 2.487 | 19.056 | 17.370 | 20.860 | 45 | 6 | 46 | 355 | 237 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p10__sea_bg_0p50__sea_cfg_0p20 | 0.10 | 0.50 | 0.20 | 539.052 | 212.991 | 2.531 | 16.158 | 15.140 | 16.920 | 45 | 6 | 46 | 361 | 231 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p10__sea_bg_0p50__sea_cfg_0p20 | 0.10 | 0.50 | 0.20 | 538.562 | 208.608 | 2.582 | 20.713 | 20.050 | 21.120 | 45 | 6 | 46 | 368 | 224 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p10__sea_bg_0p50__sea_cfg_0p20 | 0.10 | 0.50 | 0.20 | 539.330 | 221.388 | 2.436 | 18.210 | 17.890 | 18.660 | 45 | 6 | 46 | 348 | 244 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p10__sea_bg_0p50__sea_cfg_0p20 | 0.10 | 0.50 | 0.20 | 542.662 | 221.763 | 2.447 | 17.636 | 16.020 | 19.900 | 45 | 5 | 46 | 358 | 242 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p10__sea_bg_0p50__sea_cfg_0p20 | 0.10 | 0.50 | 0.20 | 544.363 | 214.983 | 2.532 | 32.019 | 29.400 | 34.640 | 45 | 6 | 46 | 361 | 231 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p10__sea_bg_0p50__sea_cfg_0p20 | 0.10 | 0.50 | 0.20 | 544.006 | 214.951 | 2.531 | 20.861 | 20.220 | 21.590 | 45 | 6 | 46 | 361 | 231 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p10__sea_bg_0p50__sea_cfg_0p20 | 0.10 | 0.50 | 0.20 | 543.652 | 214.938 | 2.529 | 19.022 | 18.550 | 19.720 | 45 | 6 | 46 | 361 | 231 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p10__sea_bg_0p50__sea_cfg_0p20 | 0.10 | 0.50 | 0.20 | 544.915 | 223.256 | 2.441 | 24.054 | 23.010 | 24.730 | 45 | 6 | 46 | 348 | 244 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p10__sea_bg_0p50__sea_cfg_0p50 | 0.10 | 0.50 | 0.50 | 537.977 | 178.726 | 3.010 | 15.158 | 13.060 | 16.460 | 45 | 6 | 45 | 272 | 192 | 36 | 14 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p10__sea_bg_0p50__sea_cfg_0p50 | 0.10 | 0.50 | 0.50 | 538.450 | 187.015 | 2.879 | 17.847 | 16.300 | 19.120 | 45 | 6 | 45 | 259 | 205 | 36 | 14 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p10__sea_bg_0p50__sea_cfg_0p50 | 0.10 | 0.50 | 0.50 | 539.052 | 183.879 | 2.932 | 15.574 | 14.440 | 16.510 | 45 | 6 | 44 | 273 | 199 | 35 | 15 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p10__sea_bg_0p50__sea_cfg_0p50 | 0.10 | 0.50 | 0.50 | 538.562 | 183.364 | 2.937 | 20.777 | 19.510 | 21.250 | 45 | 6 | 45 | 265 | 199 | 36 | 14 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p10__sea_bg_0p50__sea_cfg_0p50 | 0.10 | 0.50 | 0.50 | 539.330 | 188.413 | 2.862 | 18.231 | 17.700 | 18.690 | 45 | 6 | 44 | 266 | 206 | 35 | 15 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p10__sea_bg_0p50__sea_cfg_0p50 | 0.10 | 0.50 | 0.50 | 542.662 | 189.224 | 2.868 | 16.845 | 15.230 | 19.540 | 45 | 5 | 45 | 275 | 205 | 35 | 15 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p10__sea_bg_0p50__sea_cfg_0p50 | 0.10 | 0.50 | 0.50 | 544.363 | 188.172 | 2.893 | 28.334 | 21.660 | 30.170 | 45 | 6 | 45 | 259 | 205 | 36 | 14 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p10__sea_bg_0p50__sea_cfg_0p50 | 0.10 | 0.50 | 0.50 | 544.006 | 184.779 | 2.944 | 19.056 | 18.610 | 19.450 | 45 | 6 | 45 | 265 | 199 | 36 | 14 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p10__sea_bg_0p50__sea_cfg_0p50 | 0.10 | 0.50 | 0.50 | 543.652 | 184.863 | 2.941 | 18.218 | 17.320 | 19.020 | 45 | 6 | 45 | 265 | 199 | 36 | 14 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p10__sea_bg_0p50__sea_cfg_0p50 | 0.10 | 0.50 | 0.50 | 544.915 | 189.363 | 2.878 | 23.746 | 23.060 | 24.310 | 45 | 6 | 45 | 258 | 206 | 36 | 14 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p20__sea_bg_0p05__sea_cfg_0p05 | 0.20 | 0.05 | 0.05 | 537.977 | 374.002 | 1.438 | 20.295 | 17.530 | 23.430 | 45 | 20 | 30 | 0 | 480 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p20__sea_bg_0p05__sea_cfg_0p05 | 0.20 | 0.05 | 0.05 | 538.450 | 374.085 | 1.439 | 32.270 | 25.220 | 37.070 | 45 | 20 | 30 | 0 | 480 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p20__sea_bg_0p05__sea_cfg_0p05 | 0.20 | 0.05 | 0.05 | 539.052 | 374.814 | 1.438 | 20.250 | 17.550 | 23.470 | 45 | 20 | 30 | 0 | 480 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p20__sea_bg_0p05__sea_cfg_0p05 | 0.20 | 0.05 | 0.05 | 538.562 | 374.500 | 1.438 | 24.736 | 23.830 | 25.150 | 45 | 20 | 30 | 0 | 480 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p20__sea_bg_0p05__sea_cfg_0p05 | 0.20 | 0.05 | 0.05 | 539.330 | 374.841 | 1.439 | 21.166 | 19.340 | 21.620 | 45 | 20 | 30 | 0 | 480 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p20__sea_bg_0p05__sea_cfg_0p05 | 0.20 | 0.05 | 0.05 | 542.662 | 377.925 | 1.436 | 24.488 | 21.920 | 26.030 | 45 | 20 | 30 | 0 | 480 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p20__sea_bg_0p05__sea_cfg_0p05 | 0.20 | 0.05 | 0.05 | 544.363 | 377.362 | 1.443 | 41.091 | 36.920 | 45.600 | 45 | 20 | 30 | 0 | 480 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p20__sea_bg_0p05__sea_cfg_0p05 | 0.20 | 0.05 | 0.05 | 544.006 | 377.820 | 1.440 | 24.702 | 24.060 | 25.530 | 45 | 20 | 30 | 0 | 480 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p20__sea_bg_0p05__sea_cfg_0p05 | 0.20 | 0.05 | 0.05 | 543.652 | 378.137 | 1.438 | 20.576 | 19.200 | 21.480 | 45 | 20 | 30 | 0 | 480 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p20__sea_bg_0p05__sea_cfg_0p05 | 0.20 | 0.05 | 0.05 | 544.915 | 377.664 | 1.443 | 30.007 | 27.170 | 31.240 | 45 | 20 | 30 | 0 | 480 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p20__sea_bg_0p05__sea_cfg_0p10 | 0.20 | 0.05 | 0.10 | 537.977 | 367.041 | 1.466 | 20.377 | 17.610 | 23.600 | 45 | 21 | 32 | 0 | 472 | 6 | 44 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p20__sea_bg_0p05__sea_cfg_0p10 | 0.20 | 0.05 | 0.10 | 538.450 | 366.694 | 1.468 | 31.079 | 24.950 | 34.880 | 45 | 21 | 32 | 0 | 472 | 6 | 44 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p20__sea_bg_0p05__sea_cfg_0p10 | 0.20 | 0.05 | 0.10 | 539.052 | 367.432 | 1.467 | 20.068 | 17.420 | 23.100 | 45 | 22 | 33 | 0 | 472 | 6 | 44 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p20__sea_bg_0p05__sea_cfg_0p10 | 0.20 | 0.05 | 0.10 | 538.562 | 366.756 | 1.468 | 24.782 | 24.020 | 25.530 | 45 | 21 | 32 | 0 | 472 | 6 | 44 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p20__sea_bg_0p05__sea_cfg_0p10 | 0.20 | 0.05 | 0.10 | 539.330 | 367.238 | 1.469 | 20.996 | 19.170 | 21.470 | 45 | 22 | 33 | 0 | 472 | 6 | 44 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p20__sea_bg_0p05__sea_cfg_0p10 | 0.20 | 0.05 | 0.10 | 542.662 | 370.807 | 1.463 | 24.357 | 21.800 | 25.640 | 45 | 23 | 33 | 0 | 472 | 5 | 45 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p20__sea_bg_0p05__sea_cfg_0p10 | 0.20 | 0.05 | 0.10 | 544.363 | 369.885 | 1.472 | 40.962 | 37.170 | 45.250 | 45 | 21 | 32 | 0 | 472 | 6 | 44 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p20__sea_bg_0p05__sea_cfg_0p10 | 0.20 | 0.05 | 0.10 | 544.006 | 370.579 | 1.468 | 23.863 | 23.370 | 24.540 | 45 | 21 | 32 | 0 | 472 | 6 | 44 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p20__sea_bg_0p05__sea_cfg_0p10 | 0.20 | 0.05 | 0.10 | 543.652 | 370.269 | 1.468 | 20.473 | 19.120 | 21.370 | 45 | 22 | 33 | 0 | 472 | 6 | 44 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p20__sea_bg_0p05__sea_cfg_0p10 | 0.20 | 0.05 | 0.10 | 544.915 | 370.281 | 1.472 | 29.352 | 27.160 | 30.450 | 45 | 21 | 32 | 0 | 472 | 6 | 44 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p20__sea_bg_0p05__sea_cfg_0p20 | 0.20 | 0.05 | 0.20 | 537.977 | 355.857 | 1.512 | 20.047 | 17.290 | 22.870 | 45 | 22 | 30 | 0 | 464 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p20__sea_bg_0p05__sea_cfg_0p20 | 0.20 | 0.05 | 0.20 | 538.450 | 355.496 | 1.515 | 32.702 | 27.560 | 37.310 | 45 | 22 | 30 | 0 | 464 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p20__sea_bg_0p05__sea_cfg_0p20 | 0.20 | 0.05 | 0.20 | 539.052 | 355.845 | 1.515 | 20.074 | 17.650 | 22.780 | 45 | 22 | 30 | 0 | 464 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p20__sea_bg_0p05__sea_cfg_0p20 | 0.20 | 0.05 | 0.20 | 538.562 | 355.908 | 1.513 | 24.335 | 23.680 | 24.790 | 45 | 22 | 30 | 0 | 464 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p20__sea_bg_0p05__sea_cfg_0p20 | 0.20 | 0.05 | 0.20 | 539.330 | 356.532 | 1.513 | 20.948 | 19.310 | 21.430 | 45 | 22 | 30 | 0 | 464 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p20__sea_bg_0p05__sea_cfg_0p20 | 0.20 | 0.05 | 0.20 | 542.662 | 359.137 | 1.511 | 24.154 | 21.660 | 26.080 | 45 | 22 | 30 | 0 | 464 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p20__sea_bg_0p05__sea_cfg_0p20 | 0.20 | 0.05 | 0.20 | 544.363 | 358.742 | 1.517 | 40.446 | 35.790 | 45.200 | 45 | 22 | 30 | 0 | 464 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p20__sea_bg_0p05__sea_cfg_0p20 | 0.20 | 0.05 | 0.20 | 544.006 | 358.459 | 1.518 | 24.065 | 23.410 | 24.820 | 45 | 22 | 30 | 0 | 464 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p20__sea_bg_0p05__sea_cfg_0p20 | 0.20 | 0.05 | 0.20 | 543.652 | 359.327 | 1.513 | 20.494 | 19.120 | 21.490 | 45 | 22 | 30 | 0 | 464 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p20__sea_bg_0p05__sea_cfg_0p20 | 0.20 | 0.05 | 0.20 | 544.915 | 358.720 | 1.519 | 30.327 | 28.650 | 31.180 | 45 | 22 | 30 | 0 | 464 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p20__sea_bg_0p05__sea_cfg_0p50 | 0.20 | 0.05 | 0.50 | 537.977 | 276.183 | 1.948 | 15.531 | 13.580 | 17.070 | 45 | 20 | 35 | 0 | 352 | 36 | 14 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p20__sea_bg_0p05__sea_cfg_0p50 | 0.20 | 0.05 | 0.50 | 538.450 | 276.223 | 1.949 | 19.966 | 17.740 | 22.890 | 45 | 20 | 34 | 0 | 352 | 36 | 14 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p20__sea_bg_0p05__sea_cfg_0p50 | 0.20 | 0.05 | 0.50 | 539.052 | 282.370 | 1.909 | 15.423 | 14.450 | 16.240 | 45 | 20 | 33 | 0 | 360 | 35 | 15 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p20__sea_bg_0p05__sea_cfg_0p50 | 0.20 | 0.05 | 0.50 | 538.562 | 276.488 | 1.948 | 22.212 | 21.070 | 22.700 | 45 | 20 | 35 | 0 | 352 | 36 | 14 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p20__sea_bg_0p05__sea_cfg_0p50 | 0.20 | 0.05 | 0.50 | 539.330 | 282.548 | 1.909 | 17.843 | 17.150 | 18.160 | 45 | 20 | 33 | 0 | 360 | 35 | 15 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p20__sea_bg_0p05__sea_cfg_0p50 | 0.20 | 0.05 | 0.50 | 542.662 | 284.327 | 1.909 | 17.115 | 15.380 | 19.510 | 45 | 20 | 33 | 0 | 360 | 35 | 15 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p20__sea_bg_0p05__sea_cfg_0p50 | 0.20 | 0.05 | 0.50 | 544.363 | 278.616 | 1.954 | 32.610 | 22.540 | 35.320 | 45 | 20 | 35 | 0 | 352 | 36 | 14 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p20__sea_bg_0p05__sea_cfg_0p50 | 0.20 | 0.05 | 0.50 | 544.006 | 278.405 | 1.954 | 21.293 | 20.490 | 21.810 | 45 | 20 | 35 | 0 | 352 | 36 | 14 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p20__sea_bg_0p05__sea_cfg_0p50 | 0.20 | 0.05 | 0.50 | 543.652 | 284.652 | 1.910 | 19.766 | 18.910 | 20.460 | 45 | 20 | 33 | 0 | 360 | 35 | 15 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p20__sea_bg_0p05__sea_cfg_0p50 | 0.20 | 0.05 | 0.50 | 544.915 | 278.833 | 1.954 | 25.278 | 23.940 | 26.010 | 45 | 20 | 34 | 0 | 352 | 36 | 14 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p20__sea_bg_0p10__sea_cfg_0p05 | 0.20 | 0.10 | 0.05 | 537.977 | 374.458 | 1.437 | 20.295 | 17.530 | 23.430 | 45 | 20 | 30 | 0 | 480 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p20__sea_bg_0p10__sea_cfg_0p05 | 0.20 | 0.10 | 0.05 | 538.450 | 374.492 | 1.438 | 32.270 | 25.220 | 37.070 | 45 | 20 | 30 | 0 | 480 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p20__sea_bg_0p10__sea_cfg_0p05 | 0.20 | 0.10 | 0.05 | 539.052 | 375.431 | 1.436 | 20.250 | 17.550 | 23.470 | 45 | 20 | 30 | 0 | 480 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p20__sea_bg_0p10__sea_cfg_0p05 | 0.20 | 0.10 | 0.05 | 538.562 | 374.926 | 1.436 | 24.736 | 23.830 | 25.150 | 45 | 20 | 30 | 0 | 480 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p20__sea_bg_0p10__sea_cfg_0p05 | 0.20 | 0.10 | 0.05 | 539.330 | 374.360 | 1.441 | 21.166 | 19.340 | 21.620 | 45 | 20 | 30 | 0 | 480 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p20__sea_bg_0p10__sea_cfg_0p05 | 0.20 | 0.10 | 0.05 | 542.662 | 377.851 | 1.436 | 24.488 | 21.920 | 26.030 | 45 | 20 | 30 | 0 | 480 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p20__sea_bg_0p10__sea_cfg_0p05 | 0.20 | 0.10 | 0.05 | 544.363 | 377.796 | 1.441 | 41.091 | 36.920 | 45.600 | 45 | 20 | 30 | 0 | 480 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p20__sea_bg_0p10__sea_cfg_0p05 | 0.20 | 0.10 | 0.05 | 544.006 | 377.845 | 1.440 | 24.702 | 24.060 | 25.530 | 45 | 20 | 30 | 0 | 480 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p20__sea_bg_0p10__sea_cfg_0p05 | 0.20 | 0.10 | 0.05 | 543.652 | 378.427 | 1.437 | 20.576 | 19.200 | 21.480 | 45 | 20 | 30 | 0 | 480 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p20__sea_bg_0p10__sea_cfg_0p05 | 0.20 | 0.10 | 0.05 | 544.915 | 377.947 | 1.442 | 30.007 | 27.170 | 31.240 | 45 | 20 | 30 | 0 | 480 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p20__sea_bg_0p10__sea_cfg_0p10 | 0.20 | 0.10 | 0.10 | 537.977 | 366.972 | 1.466 | 20.377 | 17.610 | 23.600 | 45 | 21 | 32 | 0 | 472 | 6 | 44 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p20__sea_bg_0p10__sea_cfg_0p10 | 0.20 | 0.10 | 0.10 | 538.450 | 366.726 | 1.468 | 31.079 | 24.950 | 34.880 | 45 | 21 | 32 | 0 | 472 | 6 | 44 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p20__sea_bg_0p10__sea_cfg_0p10 | 0.20 | 0.10 | 0.10 | 539.052 | 367.380 | 1.467 | 20.068 | 17.420 | 23.100 | 45 | 22 | 33 | 0 | 472 | 6 | 44 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p20__sea_bg_0p10__sea_cfg_0p10 | 0.20 | 0.10 | 0.10 | 538.562 | 367.190 | 1.467 | 24.782 | 24.020 | 25.530 | 45 | 21 | 32 | 0 | 472 | 6 | 44 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p20__sea_bg_0p10__sea_cfg_0p10 | 0.20 | 0.10 | 0.10 | 539.330 | 366.895 | 1.470 | 20.996 | 19.170 | 21.470 | 45 | 22 | 33 | 0 | 472 | 6 | 44 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p20__sea_bg_0p10__sea_cfg_0p10 | 0.20 | 0.10 | 0.10 | 542.662 | 370.665 | 1.464 | 24.357 | 21.800 | 25.640 | 45 | 23 | 33 | 0 | 472 | 5 | 45 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p20__sea_bg_0p10__sea_cfg_0p10 | 0.20 | 0.10 | 0.10 | 544.363 | 370.382 | 1.470 | 40.962 | 37.170 | 45.250 | 45 | 21 | 32 | 0 | 472 | 6 | 44 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p20__sea_bg_0p10__sea_cfg_0p10 | 0.20 | 0.10 | 0.10 | 544.006 | 369.866 | 1.471 | 23.863 | 23.370 | 24.540 | 45 | 21 | 32 | 0 | 472 | 6 | 44 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p20__sea_bg_0p10__sea_cfg_0p10 | 0.20 | 0.10 | 0.10 | 543.652 | 370.488 | 1.467 | 20.473 | 19.120 | 21.370 | 45 | 22 | 33 | 0 | 472 | 6 | 44 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p20__sea_bg_0p10__sea_cfg_0p10 | 0.20 | 0.10 | 0.10 | 544.915 | 370.229 | 1.472 | 29.352 | 27.160 | 30.450 | 45 | 21 | 32 | 0 | 472 | 6 | 44 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p20__sea_bg_0p10__sea_cfg_0p20 | 0.20 | 0.10 | 0.20 | 537.977 | 356.058 | 1.511 | 20.047 | 17.290 | 22.870 | 45 | 22 | 30 | 0 | 464 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p20__sea_bg_0p10__sea_cfg_0p20 | 0.20 | 0.10 | 0.20 | 538.450 | 355.832 | 1.513 | 32.702 | 27.560 | 37.310 | 45 | 22 | 30 | 0 | 464 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p20__sea_bg_0p10__sea_cfg_0p20 | 0.20 | 0.10 | 0.20 | 539.052 | 356.590 | 1.512 | 20.074 | 17.650 | 22.780 | 45 | 22 | 30 | 0 | 464 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p20__sea_bg_0p10__sea_cfg_0p20 | 0.20 | 0.10 | 0.20 | 538.562 | 356.000 | 1.513 | 24.335 | 23.680 | 24.790 | 45 | 22 | 30 | 0 | 464 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p20__sea_bg_0p10__sea_cfg_0p20 | 0.20 | 0.10 | 0.20 | 539.330 | 355.656 | 1.516 | 20.948 | 19.310 | 21.430 | 45 | 22 | 30 | 0 | 464 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p20__sea_bg_0p10__sea_cfg_0p20 | 0.20 | 0.10 | 0.20 | 542.662 | 358.955 | 1.512 | 24.154 | 21.660 | 26.080 | 45 | 22 | 30 | 0 | 464 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p20__sea_bg_0p10__sea_cfg_0p20 | 0.20 | 0.10 | 0.20 | 544.363 | 358.928 | 1.517 | 40.446 | 35.790 | 45.200 | 45 | 22 | 30 | 0 | 464 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p20__sea_bg_0p10__sea_cfg_0p20 | 0.20 | 0.10 | 0.20 | 544.006 | 358.844 | 1.516 | 24.065 | 23.410 | 24.820 | 45 | 22 | 30 | 0 | 464 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p20__sea_bg_0p10__sea_cfg_0p20 | 0.20 | 0.10 | 0.20 | 543.652 | 359.620 | 1.512 | 20.494 | 19.120 | 21.490 | 45 | 22 | 30 | 0 | 464 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p20__sea_bg_0p10__sea_cfg_0p20 | 0.20 | 0.10 | 0.20 | 544.915 | 359.086 | 1.518 | 30.327 | 28.650 | 31.180 | 45 | 22 | 30 | 0 | 464 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p20__sea_bg_0p10__sea_cfg_0p50 | 0.20 | 0.10 | 0.50 | 537.977 | 276.327 | 1.947 | 15.531 | 13.580 | 17.070 | 45 | 20 | 35 | 0 | 352 | 36 | 14 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p20__sea_bg_0p10__sea_cfg_0p50 | 0.20 | 0.10 | 0.50 | 538.450 | 276.424 | 1.948 | 19.966 | 17.740 | 22.890 | 45 | 20 | 34 | 0 | 352 | 36 | 14 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p20__sea_bg_0p10__sea_cfg_0p50 | 0.20 | 0.10 | 0.50 | 539.052 | 281.867 | 1.912 | 15.423 | 14.450 | 16.240 | 45 | 20 | 33 | 0 | 360 | 35 | 15 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p20__sea_bg_0p10__sea_cfg_0p50 | 0.20 | 0.10 | 0.50 | 538.562 | 276.297 | 1.949 | 22.212 | 21.070 | 22.700 | 45 | 20 | 35 | 0 | 352 | 36 | 14 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p20__sea_bg_0p10__sea_cfg_0p50 | 0.20 | 0.10 | 0.50 | 539.330 | 281.740 | 1.914 | 17.843 | 17.150 | 18.160 | 45 | 20 | 33 | 0 | 360 | 35 | 15 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p20__sea_bg_0p10__sea_cfg_0p50 | 0.20 | 0.10 | 0.50 | 542.662 | 284.505 | 1.907 | 17.115 | 15.380 | 19.510 | 45 | 20 | 33 | 0 | 360 | 35 | 15 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p20__sea_bg_0p10__sea_cfg_0p50 | 0.20 | 0.10 | 0.50 | 544.363 | 278.747 | 1.953 | 32.610 | 22.540 | 35.320 | 45 | 20 | 35 | 0 | 352 | 36 | 14 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p20__sea_bg_0p10__sea_cfg_0p50 | 0.20 | 0.10 | 0.50 | 544.006 | 278.739 | 1.952 | 21.293 | 20.490 | 21.810 | 45 | 20 | 35 | 0 | 352 | 36 | 14 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p20__sea_bg_0p10__sea_cfg_0p50 | 0.20 | 0.10 | 0.50 | 543.652 | 284.836 | 1.909 | 19.766 | 18.910 | 20.460 | 45 | 20 | 33 | 0 | 360 | 35 | 15 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p20__sea_bg_0p10__sea_cfg_0p50 | 0.20 | 0.10 | 0.50 | 544.915 | 278.912 | 1.954 | 25.278 | 23.940 | 26.010 | 45 | 20 | 34 | 0 | 352 | 36 | 14 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p20__sea_bg_0p20__sea_cfg_0p05 | 0.20 | 0.20 | 0.05 | 537.977 | 371.728 | 1.447 | 20.290 | 17.510 | 23.250 | 45 | 20 | 30 | 5 | 475 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p20__sea_bg_0p20__sea_cfg_0p05 | 0.20 | 0.20 | 0.05 | 538.450 | 372.156 | 1.447 | 31.941 | 25.210 | 36.700 | 45 | 20 | 30 | 4 | 476 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p20__sea_bg_0p20__sea_cfg_0p05 | 0.20 | 0.20 | 0.05 | 539.052 | 372.011 | 1.449 | 20.349 | 17.560 | 23.720 | 45 | 20 | 30 | 4 | 476 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p20__sea_bg_0p20__sea_cfg_0p05 | 0.20 | 0.20 | 0.05 | 538.562 | 372.494 | 1.446 | 24.766 | 24.120 | 25.230 | 45 | 20 | 30 | 4 | 476 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p20__sea_bg_0p20__sea_cfg_0p05 | 0.20 | 0.20 | 0.05 | 539.330 | 371.393 | 1.452 | 20.116 | 19.220 | 20.590 | 45 | 20 | 30 | 5 | 475 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p20__sea_bg_0p20__sea_cfg_0p05 | 0.20 | 0.20 | 0.05 | 542.662 | 375.846 | 1.444 | 24.410 | 21.610 | 26.620 | 45 | 20 | 30 | 4 | 476 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p20__sea_bg_0p20__sea_cfg_0p05 | 0.20 | 0.20 | 0.05 | 544.363 | 375.450 | 1.450 | 40.962 | 36.550 | 45.760 | 45 | 20 | 30 | 4 | 476 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p20__sea_bg_0p20__sea_cfg_0p05 | 0.20 | 0.20 | 0.05 | 544.006 | 375.198 | 1.450 | 23.724 | 23.140 | 24.340 | 45 | 20 | 30 | 4 | 476 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p20__sea_bg_0p20__sea_cfg_0p05 | 0.20 | 0.20 | 0.05 | 543.652 | 375.321 | 1.448 | 20.550 | 19.170 | 21.490 | 45 | 20 | 30 | 5 | 475 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p20__sea_bg_0p20__sea_cfg_0p05 | 0.20 | 0.20 | 0.05 | 544.915 | 374.115 | 1.457 | 29.404 | 26.600 | 30.370 | 45 | 20 | 30 | 6 | 474 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p20__sea_bg_0p20__sea_cfg_0p10 | 0.20 | 0.20 | 0.10 | 537.977 | 364.237 | 1.477 | 20.387 | 17.550 | 23.490 | 45 | 21 | 32 | 5 | 467 | 6 | 44 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p20__sea_bg_0p20__sea_cfg_0p10 | 0.20 | 0.20 | 0.10 | 538.450 | 364.630 | 1.477 | 30.765 | 24.940 | 34.500 | 45 | 21 | 32 | 4 | 468 | 6 | 44 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p20__sea_bg_0p20__sea_cfg_0p10 | 0.20 | 0.20 | 0.10 | 539.052 | 364.310 | 1.480 | 20.078 | 17.380 | 23.250 | 45 | 22 | 33 | 4 | 468 | 6 | 44 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p20__sea_bg_0p20__sea_cfg_0p10 | 0.20 | 0.20 | 0.10 | 538.562 | 365.001 | 1.476 | 24.746 | 24.140 | 25.360 | 45 | 21 | 32 | 4 | 468 | 6 | 44 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p20__sea_bg_0p20__sea_cfg_0p10 | 0.20 | 0.20 | 0.10 | 539.330 | 363.887 | 1.482 | 20.006 | 19.110 | 20.480 | 45 | 22 | 33 | 5 | 467 | 6 | 44 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p20__sea_bg_0p20__sea_cfg_0p10 | 0.20 | 0.20 | 0.10 | 542.662 | 368.214 | 1.474 | 24.178 | 21.450 | 26.230 | 45 | 23 | 33 | 4 | 468 | 5 | 45 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p20__sea_bg_0p20__sea_cfg_0p10 | 0.20 | 0.20 | 0.10 | 544.363 | 367.855 | 1.480 | 40.557 | 36.540 | 45.060 | 45 | 21 | 32 | 4 | 468 | 6 | 44 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p20__sea_bg_0p20__sea_cfg_0p10 | 0.20 | 0.20 | 0.10 | 544.006 | 367.692 | 1.480 | 23.406 | 22.820 | 23.940 | 45 | 21 | 32 | 4 | 468 | 6 | 44 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p20__sea_bg_0p20__sea_cfg_0p10 | 0.20 | 0.20 | 0.10 | 543.652 | 367.713 | 1.478 | 20.482 | 19.120 | 21.360 | 45 | 22 | 33 | 5 | 467 | 6 | 44 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p20__sea_bg_0p20__sea_cfg_0p10 | 0.20 | 0.20 | 0.10 | 544.915 | 366.392 | 1.487 | 28.807 | 26.590 | 29.720 | 45 | 21 | 32 | 6 | 466 | 6 | 44 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p20__sea_bg_0p20__sea_cfg_0p20 | 0.20 | 0.20 | 0.20 | 537.977 | 354.490 | 1.518 | 20.223 | 17.120 | 23.810 | 45 | 22 | 30 | 2 | 462 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p20__sea_bg_0p20__sea_cfg_0p20 | 0.20 | 0.20 | 0.20 | 538.450 | 354.671 | 1.518 | 32.564 | 27.380 | 37.040 | 45 | 22 | 30 | 2 | 462 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p20__sea_bg_0p20__sea_cfg_0p20 | 0.20 | 0.20 | 0.20 | 539.052 | 354.546 | 1.520 | 20.227 | 17.750 | 23.110 | 45 | 22 | 30 | 2 | 462 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p20__sea_bg_0p20__sea_cfg_0p20 | 0.20 | 0.20 | 0.20 | 538.562 | 355.166 | 1.516 | 24.487 | 23.780 | 24.900 | 45 | 22 | 30 | 2 | 462 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p20__sea_bg_0p20__sea_cfg_0p20 | 0.20 | 0.20 | 0.20 | 539.330 | 353.906 | 1.524 | 20.057 | 19.270 | 20.460 | 45 | 22 | 30 | 3 | 461 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p20__sea_bg_0p20__sea_cfg_0p20 | 0.20 | 0.20 | 0.20 | 542.662 | 357.826 | 1.517 | 24.159 | 21.460 | 25.810 | 45 | 22 | 30 | 2 | 462 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p20__sea_bg_0p20__sea_cfg_0p20 | 0.20 | 0.20 | 0.20 | 544.363 | 357.818 | 1.521 | 40.341 | 36.340 | 45.230 | 45 | 22 | 30 | 2 | 462 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p20__sea_bg_0p20__sea_cfg_0p20 | 0.20 | 0.20 | 0.20 | 544.006 | 357.615 | 1.521 | 23.223 | 22.670 | 23.830 | 45 | 22 | 30 | 2 | 462 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p20__sea_bg_0p20__sea_cfg_0p20 | 0.20 | 0.20 | 0.20 | 543.652 | 358.440 | 1.517 | 20.567 | 19.180 | 21.560 | 45 | 22 | 30 | 2 | 462 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p20__sea_bg_0p20__sea_cfg_0p20 | 0.20 | 0.20 | 0.20 | 544.915 | 356.751 | 1.527 | 29.247 | 27.200 | 30.100 | 45 | 22 | 30 | 4 | 460 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p20__sea_bg_0p20__sea_cfg_0p50 | 0.20 | 0.20 | 0.50 | 537.977 | 274.609 | 1.959 | 15.585 | 13.610 | 17.090 | 45 | 20 | 35 | 3 | 349 | 36 | 14 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p20__sea_bg_0p20__sea_cfg_0p50 | 0.20 | 0.20 | 0.50 | 538.450 | 275.182 | 1.957 | 19.980 | 17.760 | 22.910 | 45 | 20 | 34 | 2 | 350 | 36 | 14 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p20__sea_bg_0p20__sea_cfg_0p50 | 0.20 | 0.20 | 0.50 | 539.052 | 280.834 | 1.919 | 15.449 | 14.530 | 16.250 | 45 | 20 | 33 | 2 | 358 | 35 | 15 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p20__sea_bg_0p20__sea_cfg_0p50 | 0.20 | 0.20 | 0.50 | 538.562 | 275.678 | 1.954 | 22.311 | 21.220 | 22.760 | 45 | 20 | 35 | 2 | 350 | 36 | 14 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p20__sea_bg_0p20__sea_cfg_0p50 | 0.20 | 0.20 | 0.50 | 539.330 | 280.770 | 1.921 | 17.898 | 17.210 | 18.220 | 45 | 20 | 33 | 2 | 358 | 35 | 15 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p20__sea_bg_0p20__sea_cfg_0p50 | 0.20 | 0.20 | 0.50 | 542.662 | 283.342 | 1.915 | 17.135 | 15.420 | 19.480 | 45 | 20 | 33 | 2 | 358 | 35 | 15 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p20__sea_bg_0p20__sea_cfg_0p50 | 0.20 | 0.20 | 0.50 | 544.363 | 277.723 | 1.960 | 32.650 | 22.560 | 35.310 | 45 | 20 | 35 | 2 | 350 | 36 | 14 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p20__sea_bg_0p20__sea_cfg_0p50 | 0.20 | 0.20 | 0.50 | 544.006 | 277.515 | 1.960 | 21.321 | 20.520 | 21.840 | 45 | 20 | 35 | 2 | 350 | 36 | 14 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p20__sea_bg_0p20__sea_cfg_0p50 | 0.20 | 0.20 | 0.50 | 543.652 | 283.560 | 1.917 | 19.746 | 18.910 | 20.420 | 45 | 20 | 33 | 2 | 358 | 35 | 15 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p20__sea_bg_0p20__sea_cfg_0p50 | 0.20 | 0.20 | 0.50 | 544.915 | 276.961 | 1.967 | 25.173 | 23.770 | 25.810 | 45 | 20 | 34 | 3 | 349 | 36 | 14 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p20__sea_bg_0p50__sea_cfg_0p05 | 0.20 | 0.50 | 0.05 | 537.977 | 214.583 | 2.507 | 15.576 | 14.000 | 17.230 | 45 | 20 | 30 | 256 | 224 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p20__sea_bg_0p50__sea_cfg_0p05 | 0.20 | 0.50 | 0.05 | 538.450 | 214.716 | 2.508 | 18.725 | 16.890 | 20.820 | 45 | 20 | 30 | 256 | 224 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p20__sea_bg_0p50__sea_cfg_0p05 | 0.20 | 0.50 | 0.05 | 539.052 | 214.781 | 2.510 | 15.731 | 15.170 | 16.710 | 45 | 20 | 30 | 256 | 224 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p20__sea_bg_0p50__sea_cfg_0p05 | 0.20 | 0.50 | 0.05 | 538.562 | 214.536 | 2.510 | 21.642 | 20.720 | 22.080 | 45 | 20 | 30 | 256 | 224 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p20__sea_bg_0p50__sea_cfg_0p05 | 0.20 | 0.50 | 0.05 | 539.330 | 222.161 | 2.428 | 18.079 | 17.620 | 18.500 | 45 | 20 | 30 | 244 | 236 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p20__sea_bg_0p50__sea_cfg_0p05 | 0.20 | 0.50 | 0.05 | 542.662 | 223.179 | 2.432 | 17.194 | 15.540 | 19.360 | 45 | 20 | 30 | 245 | 235 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p20__sea_bg_0p50__sea_cfg_0p05 | 0.20 | 0.50 | 0.05 | 544.363 | 216.460 | 2.515 | 32.680 | 29.200 | 35.760 | 45 | 20 | 30 | 256 | 224 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p20__sea_bg_0p50__sea_cfg_0p05 | 0.20 | 0.50 | 0.05 | 544.006 | 216.414 | 2.514 | 21.169 | 20.590 | 21.880 | 45 | 20 | 30 | 256 | 224 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p20__sea_bg_0p50__sea_cfg_0p05 | 0.20 | 0.50 | 0.05 | 543.652 | 216.619 | 2.510 | 19.483 | 18.880 | 20.140 | 45 | 20 | 30 | 256 | 224 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p20__sea_bg_0p50__sea_cfg_0p05 | 0.20 | 0.50 | 0.05 | 544.915 | 224.045 | 2.432 | 24.255 | 23.190 | 24.770 | 45 | 20 | 30 | 244 | 236 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p20__sea_bg_0p50__sea_cfg_0p10 | 0.20 | 0.50 | 0.10 | 537.977 | 211.573 | 2.543 | 15.637 | 14.050 | 17.200 | 45 | 21 | 32 | 248 | 224 | 6 | 44 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p20__sea_bg_0p50__sea_cfg_0p10 | 0.20 | 0.50 | 0.10 | 538.450 | 211.921 | 2.541 | 18.745 | 16.850 | 20.790 | 45 | 21 | 32 | 248 | 224 | 6 | 44 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p20__sea_bg_0p50__sea_cfg_0p10 | 0.20 | 0.50 | 0.10 | 539.052 | 207.044 | 2.604 | 15.456 | 14.740 | 16.510 | 45 | 22 | 33 | 256 | 216 | 6 | 44 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p20__sea_bg_0p50__sea_cfg_0p10 | 0.20 | 0.50 | 0.10 | 538.562 | 211.687 | 2.544 | 20.859 | 20.010 | 21.190 | 45 | 21 | 32 | 248 | 224 | 6 | 44 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p20__sea_bg_0p50__sea_cfg_0p10 | 0.20 | 0.50 | 0.10 | 539.330 | 214.396 | 2.516 | 17.812 | 17.330 | 18.270 | 45 | 22 | 33 | 244 | 228 | 6 | 44 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p20__sea_bg_0p50__sea_cfg_0p10 | 0.20 | 0.50 | 0.10 | 542.662 | 220.892 | 2.457 | 17.108 | 15.480 | 19.170 | 45 | 23 | 33 | 237 | 235 | 5 | 45 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p20__sea_bg_0p50__sea_cfg_0p10 | 0.20 | 0.50 | 0.10 | 544.363 | 213.916 | 2.545 | 31.389 | 28.710 | 33.600 | 45 | 21 | 32 | 248 | 224 | 6 | 44 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p20__sea_bg_0p50__sea_cfg_0p10 | 0.20 | 0.50 | 0.10 | 544.006 | 213.566 | 2.547 | 21.075 | 20.440 | 21.750 | 45 | 21 | 32 | 248 | 224 | 6 | 44 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p20__sea_bg_0p50__sea_cfg_0p10 | 0.20 | 0.50 | 0.10 | 543.652 | 208.928 | 2.602 | 19.192 | 18.550 | 19.840 | 45 | 22 | 33 | 256 | 216 | 6 | 44 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p20__sea_bg_0p50__sea_cfg_0p10 | 0.20 | 0.50 | 0.10 | 544.915 | 221.282 | 2.463 | 24.188 | 23.210 | 24.660 | 45 | 21 | 32 | 236 | 236 | 6 | 44 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p20__sea_bg_0p50__sea_cfg_0p20 | 0.20 | 0.50 | 0.20 | 537.977 | 200.543 | 2.683 | 15.428 | 13.750 | 16.690 | 45 | 22 | 30 | 248 | 216 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p20__sea_bg_0p50__sea_cfg_0p20 | 0.20 | 0.50 | 0.20 | 538.450 | 200.519 | 2.685 | 18.651 | 16.910 | 20.610 | 45 | 22 | 30 | 248 | 216 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p20__sea_bg_0p50__sea_cfg_0p20 | 0.20 | 0.50 | 0.20 | 539.052 | 200.750 | 2.685 | 15.638 | 14.920 | 16.830 | 45 | 22 | 30 | 248 | 216 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p20__sea_bg_0p50__sea_cfg_0p20 | 0.20 | 0.50 | 0.20 | 538.562 | 200.688 | 2.684 | 21.641 | 20.820 | 22.000 | 45 | 22 | 30 | 248 | 216 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p20__sea_bg_0p50__sea_cfg_0p20 | 0.20 | 0.50 | 0.20 | 539.330 | 208.446 | 2.587 | 18.098 | 17.600 | 18.520 | 45 | 22 | 30 | 236 | 228 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p20__sea_bg_0p50__sea_cfg_0p20 | 0.20 | 0.50 | 0.20 | 542.662 | 209.033 | 2.596 | 17.057 | 15.440 | 19.270 | 45 | 22 | 30 | 237 | 227 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p20__sea_bg_0p50__sea_cfg_0p20 | 0.20 | 0.50 | 0.20 | 544.363 | 202.293 | 2.691 | 33.072 | 29.840 | 36.340 | 45 | 22 | 30 | 248 | 216 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p20__sea_bg_0p50__sea_cfg_0p20 | 0.20 | 0.50 | 0.20 | 544.006 | 202.345 | 2.689 | 21.160 | 20.540 | 21.750 | 45 | 22 | 30 | 248 | 216 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p20__sea_bg_0p50__sea_cfg_0p20 | 0.20 | 0.50 | 0.20 | 543.652 | 202.657 | 2.683 | 19.243 | 18.590 | 19.840 | 45 | 22 | 30 | 248 | 216 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p20__sea_bg_0p50__sea_cfg_0p20 | 0.20 | 0.50 | 0.20 | 544.915 | 209.962 | 2.595 | 24.170 | 23.010 | 24.630 | 45 | 22 | 30 | 236 | 228 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p20__sea_bg_0p50__sea_cfg_0p50 | 0.20 | 0.50 | 0.50 | 537.977 | 176.192 | 3.053 | 16.083 | 14.490 | 17.470 | 45 | 20 | 35 | 160 | 192 | 36 | 14 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p20__sea_bg_0p50__sea_cfg_0p50 | 0.20 | 0.50 | 0.50 | 538.450 | 176.192 | 3.056 | 17.865 | 16.250 | 19.200 | 45 | 20 | 34 | 160 | 192 | 36 | 14 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p20__sea_bg_0p50__sea_cfg_0p50 | 0.20 | 0.50 | 0.50 | 539.052 | 177.085 | 3.044 | 16.096 | 14.800 | 16.970 | 45 | 20 | 33 | 168 | 192 | 35 | 15 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p20__sea_bg_0p50__sea_cfg_0p50 | 0.20 | 0.50 | 0.50 | 538.562 | 176.098 | 3.058 | 20.293 | 19.810 | 20.810 | 45 | 20 | 35 | 160 | 192 | 36 | 14 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p20__sea_bg_0p50__sea_cfg_0p50 | 0.20 | 0.50 | 0.50 | 539.330 | 180.672 | 2.985 | 18.099 | 17.660 | 18.630 | 45 | 20 | 33 | 162 | 198 | 35 | 15 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p20__sea_bg_0p50__sea_cfg_0p50 | 0.20 | 0.50 | 0.50 | 542.662 | 181.875 | 2.984 | 16.796 | 15.590 | 19.610 | 45 | 20 | 33 | 162 | 198 | 35 | 15 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p20__sea_bg_0p50__sea_cfg_0p50 | 0.20 | 0.50 | 0.50 | 544.363 | 177.704 | 3.063 | 30.543 | 27.750 | 33.270 | 45 | 20 | 35 | 160 | 192 | 36 | 14 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p20__sea_bg_0p50__sea_cfg_0p50 | 0.20 | 0.50 | 0.50 | 544.006 | 177.873 | 3.058 | 19.247 | 18.890 | 19.480 | 45 | 20 | 35 | 160 | 192 | 36 | 14 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p20__sea_bg_0p50__sea_cfg_0p50 | 0.20 | 0.50 | 0.50 | 543.652 | 178.375 | 3.048 | 18.286 | 17.160 | 18.850 | 45 | 20 | 33 | 168 | 192 | 35 | 15 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p20__sea_bg_0p50__sea_cfg_0p50 | 0.20 | 0.50 | 0.50 | 544.915 | 182.343 | 2.988 | 23.273 | 22.410 | 23.870 | 45 | 20 | 34 | 153 | 199 | 36 | 14 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p50__sea_bg_0p05__sea_cfg_0p05 | 0.50 | 0.05 | 0.05 | 537.977 | 218.678 | 2.460 | 15.582 | 13.690 | 16.770 | 45 | 35 | 15 | 0 | 240 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p50__sea_bg_0p05__sea_cfg_0p05 | 0.50 | 0.05 | 0.05 | 538.450 | 218.533 | 2.464 | 19.977 | 18.770 | 21.640 | 45 | 35 | 15 | 0 | 240 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p50__sea_bg_0p05__sea_cfg_0p05 | 0.50 | 0.05 | 0.05 | 539.052 | 229.453 | 2.349 | 16.490 | 15.290 | 17.790 | 45 | 34 | 16 | 0 | 256 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p50__sea_bg_0p05__sea_cfg_0p05 | 0.50 | 0.05 | 0.05 | 538.562 | 218.732 | 2.462 | 22.051 | 20.950 | 22.520 | 45 | 35 | 15 | 0 | 240 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p50__sea_bg_0p05__sea_cfg_0p05 | 0.50 | 0.05 | 0.05 | 539.330 | 229.506 | 2.350 | 18.471 | 17.800 | 18.930 | 45 | 34 | 16 | 0 | 256 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p50__sea_bg_0p05__sea_cfg_0p05 | 0.50 | 0.05 | 0.05 | 542.662 | 230.871 | 2.350 | 18.148 | 16.760 | 19.420 | 45 | 34 | 16 | 0 | 256 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p50__sea_bg_0p05__sea_cfg_0p05 | 0.50 | 0.05 | 0.05 | 544.363 | 220.910 | 2.464 | 33.678 | 28.840 | 36.450 | 45 | 35 | 15 | 0 | 240 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p50__sea_bg_0p05__sea_cfg_0p05 | 0.50 | 0.05 | 0.05 | 544.006 | 220.748 | 2.464 | 18.941 | 18.030 | 20.040 | 45 | 35 | 15 | 0 | 240 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p50__sea_bg_0p05__sea_cfg_0p05 | 0.50 | 0.05 | 0.05 | 543.652 | 231.169 | 2.352 | 18.963 | 17.800 | 19.810 | 45 | 34 | 16 | 0 | 256 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p50__sea_bg_0p05__sea_cfg_0p05 | 0.50 | 0.05 | 0.05 | 544.915 | 221.002 | 2.466 | 24.487 | 23.160 | 25.250 | 45 | 35 | 15 | 0 | 240 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p50__sea_bg_0p05__sea_cfg_0p10 | 0.50 | 0.05 | 0.10 | 537.977 | 216.294 | 2.487 | 15.669 | 13.710 | 16.880 | 45 | 35 | 16 | 0 | 240 | 6 | 44 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p50__sea_bg_0p05__sea_cfg_0p10 | 0.50 | 0.05 | 0.10 | 538.450 | 216.478 | 2.487 | 19.899 | 18.700 | 21.540 | 45 | 36 | 17 | 0 | 240 | 5 | 45 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p50__sea_bg_0p05__sea_cfg_0p10 | 0.50 | 0.05 | 0.10 | 539.052 | 221.875 | 2.430 | 16.307 | 14.920 | 17.620 | 45 | 37 | 18 | 0 | 248 | 5 | 45 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p50__sea_bg_0p05__sea_cfg_0p10 | 0.50 | 0.05 | 0.10 | 538.562 | 216.252 | 2.490 | 21.885 | 20.950 | 22.320 | 45 | 35 | 16 | 0 | 240 | 6 | 44 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p50__sea_bg_0p05__sea_cfg_0p10 | 0.50 | 0.05 | 0.10 | 539.330 | 222.211 | 2.427 | 18.162 | 17.510 | 18.770 | 45 | 37 | 18 | 0 | 248 | 5 | 45 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p50__sea_bg_0p05__sea_cfg_0p10 | 0.50 | 0.05 | 0.10 | 542.662 | 223.611 | 2.427 | 17.984 | 16.710 | 19.300 | 45 | 37 | 18 | 0 | 248 | 5 | 45 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p50__sea_bg_0p05__sea_cfg_0p10 | 0.50 | 0.05 | 0.10 | 544.363 | 218.325 | 2.493 | 32.918 | 28.640 | 35.680 | 45 | 36 | 16 | 0 | 240 | 6 | 44 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p50__sea_bg_0p05__sea_cfg_0p10 | 0.50 | 0.05 | 0.10 | 544.006 | 218.541 | 2.489 | 18.955 | 18.040 | 20.060 | 45 | 36 | 17 | 0 | 240 | 5 | 45 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p50__sea_bg_0p05__sea_cfg_0p10 | 0.50 | 0.05 | 0.10 | 543.652 | 223.959 | 2.427 | 18.718 | 17.650 | 19.510 | 45 | 37 | 18 | 0 | 248 | 5 | 45 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p50__sea_bg_0p05__sea_cfg_0p10 | 0.50 | 0.05 | 0.10 | 544.915 | 218.671 | 2.492 | 23.882 | 22.820 | 24.540 | 45 | 36 | 17 | 0 | 240 | 5 | 45 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p50__sea_bg_0p05__sea_cfg_0p20 | 0.50 | 0.05 | 0.20 | 537.977 | 210.498 | 2.556 | 16.415 | 14.510 | 17.930 | 45 | 36 | 21 | 0 | 240 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p50__sea_bg_0p05__sea_cfg_0p20 | 0.50 | 0.05 | 0.20 | 538.450 | 210.434 | 2.559 | 19.575 | 18.570 | 21.140 | 45 | 36 | 21 | 0 | 240 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p50__sea_bg_0p05__sea_cfg_0p20 | 0.50 | 0.05 | 0.20 | 539.052 | 215.465 | 2.502 | 16.462 | 14.940 | 17.930 | 45 | 37 | 21 | 0 | 248 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p50__sea_bg_0p05__sea_cfg_0p20 | 0.50 | 0.05 | 0.20 | 538.562 | 210.583 | 2.557 | 20.535 | 19.840 | 20.960 | 45 | 36 | 21 | 0 | 240 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p50__sea_bg_0p05__sea_cfg_0p20 | 0.50 | 0.05 | 0.20 | 539.330 | 215.607 | 2.501 | 18.194 | 17.610 | 18.660 | 45 | 37 | 21 | 0 | 248 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p50__sea_bg_0p05__sea_cfg_0p20 | 0.50 | 0.05 | 0.20 | 542.662 | 217.096 | 2.500 | 18.042 | 17.070 | 19.430 | 45 | 38 | 22 | 0 | 248 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p50__sea_bg_0p05__sea_cfg_0p20 | 0.50 | 0.05 | 0.20 | 544.363 | 217.563 | 2.502 | 33.123 | 30.130 | 36.120 | 45 | 35 | 21 | 0 | 248 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p50__sea_bg_0p05__sea_cfg_0p20 | 0.50 | 0.05 | 0.20 | 544.006 | 212.161 | 2.564 | 19.078 | 18.050 | 20.200 | 45 | 36 | 21 | 0 | 240 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p50__sea_bg_0p05__sea_cfg_0p20 | 0.50 | 0.05 | 0.20 | 543.652 | 217.749 | 2.497 | 19.041 | 18.450 | 19.840 | 45 | 37 | 21 | 0 | 248 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p50__sea_bg_0p05__sea_cfg_0p20 | 0.50 | 0.05 | 0.20 | 544.915 | 212.409 | 2.565 | 22.408 | 21.340 | 23.060 | 45 | 37 | 21 | 0 | 240 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p50__sea_bg_0p05__sea_cfg_0p50 | 0.50 | 0.05 | 0.50 | 537.977 | 177.628 | 3.029 | 14.688 | 12.650 | 16.100 | 45 | 39 | 15 | 0 | 200 | 36 | 14 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p50__sea_bg_0p05__sea_cfg_0p50 | 0.50 | 0.05 | 0.50 | 538.450 | 183.356 | 2.937 | 16.922 | 15.660 | 18.020 | 45 | 39 | 16 | 0 | 208 | 35 | 15 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p50__sea_bg_0p05__sea_cfg_0p50 | 0.50 | 0.05 | 0.50 | 539.052 | 183.269 | 2.941 | 15.210 | 14.190 | 16.110 | 45 | 39 | 16 | 0 | 208 | 35 | 15 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p50__sea_bg_0p05__sea_cfg_0p50 | 0.50 | 0.05 | 0.50 | 538.562 | 177.791 | 3.029 | 20.294 | 18.920 | 20.900 | 45 | 39 | 15 | 0 | 200 | 36 | 14 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p50__sea_bg_0p05__sea_cfg_0p50 | 0.50 | 0.05 | 0.50 | 539.330 | 183.354 | 2.941 | 17.690 | 16.730 | 18.180 | 45 | 39 | 16 | 0 | 208 | 35 | 15 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p50__sea_bg_0p05__sea_cfg_0p50 | 0.50 | 0.05 | 0.50 | 542.662 | 184.756 | 2.937 | 16.314 | 14.290 | 19.220 | 45 | 39 | 16 | 0 | 208 | 35 | 15 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p50__sea_bg_0p05__sea_cfg_0p50 | 0.50 | 0.05 | 0.50 | 544.363 | 179.169 | 3.038 | 27.607 | 25.910 | 29.380 | 45 | 39 | 15 | 0 | 200 | 36 | 14 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p50__sea_bg_0p05__sea_cfg_0p50 | 0.50 | 0.05 | 0.50 | 544.006 | 179.291 | 3.034 | 18.733 | 18.110 | 19.280 | 45 | 39 | 15 | 0 | 200 | 36 | 14 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p50__sea_bg_0p05__sea_cfg_0p50 | 0.50 | 0.05 | 0.50 | 543.652 | 184.748 | 2.943 | 17.795 | 16.980 | 18.460 | 45 | 39 | 16 | 0 | 208 | 35 | 15 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p50__sea_bg_0p05__sea_cfg_0p50 | 0.50 | 0.05 | 0.50 | 544.915 | 179.426 | 3.037 | 22.889 | 22.250 | 23.280 | 45 | 39 | 15 | 0 | 200 | 36 | 14 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p50__sea_bg_0p10__sea_cfg_0p05 | 0.50 | 0.10 | 0.05 | 537.977 | 218.781 | 2.459 | 15.582 | 13.690 | 16.770 | 45 | 35 | 15 | 0 | 240 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p50__sea_bg_0p10__sea_cfg_0p05 | 0.50 | 0.10 | 0.05 | 538.450 | 218.659 | 2.463 | 19.977 | 18.770 | 21.640 | 45 | 35 | 15 | 0 | 240 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p50__sea_bg_0p10__sea_cfg_0p05 | 0.50 | 0.10 | 0.05 | 539.052 | 229.213 | 2.352 | 16.490 | 15.290 | 17.790 | 45 | 34 | 16 | 0 | 256 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p50__sea_bg_0p10__sea_cfg_0p05 | 0.50 | 0.10 | 0.05 | 538.562 | 219.090 | 2.458 | 22.051 | 20.950 | 22.520 | 45 | 35 | 15 | 0 | 240 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p50__sea_bg_0p10__sea_cfg_0p05 | 0.50 | 0.10 | 0.05 | 539.330 | 228.946 | 2.356 | 18.471 | 17.800 | 18.930 | 45 | 34 | 16 | 0 | 256 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p50__sea_bg_0p10__sea_cfg_0p05 | 0.50 | 0.10 | 0.05 | 542.662 | 230.803 | 2.351 | 18.148 | 16.760 | 19.420 | 45 | 34 | 16 | 0 | 256 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p50__sea_bg_0p10__sea_cfg_0p05 | 0.50 | 0.10 | 0.05 | 544.363 | 220.809 | 2.465 | 33.678 | 28.840 | 36.450 | 45 | 35 | 15 | 0 | 240 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p50__sea_bg_0p10__sea_cfg_0p05 | 0.50 | 0.10 | 0.05 | 544.006 | 220.802 | 2.464 | 18.941 | 18.030 | 20.040 | 45 | 35 | 15 | 0 | 240 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p50__sea_bg_0p10__sea_cfg_0p05 | 0.50 | 0.10 | 0.05 | 543.652 | 231.428 | 2.349 | 18.963 | 17.800 | 19.810 | 45 | 34 | 16 | 0 | 256 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p50__sea_bg_0p10__sea_cfg_0p05 | 0.50 | 0.10 | 0.05 | 544.915 | 220.394 | 2.472 | 24.487 | 23.160 | 25.250 | 45 | 35 | 15 | 0 | 240 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p50__sea_bg_0p10__sea_cfg_0p10 | 0.50 | 0.10 | 0.10 | 537.977 | 216.288 | 2.487 | 15.669 | 13.710 | 16.880 | 45 | 35 | 16 | 0 | 240 | 6 | 44 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p50__sea_bg_0p10__sea_cfg_0p10 | 0.50 | 0.10 | 0.10 | 538.450 | 216.729 | 2.484 | 19.899 | 18.700 | 21.540 | 45 | 36 | 17 | 0 | 240 | 5 | 45 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p50__sea_bg_0p10__sea_cfg_0p10 | 0.50 | 0.10 | 0.10 | 539.052 | 221.986 | 2.428 | 16.307 | 14.920 | 17.620 | 45 | 37 | 18 | 0 | 248 | 5 | 45 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p50__sea_bg_0p10__sea_cfg_0p10 | 0.50 | 0.10 | 0.10 | 538.562 | 216.410 | 2.489 | 21.885 | 20.950 | 22.320 | 45 | 35 | 16 | 0 | 240 | 6 | 44 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p50__sea_bg_0p10__sea_cfg_0p10 | 0.50 | 0.10 | 0.10 | 539.330 | 221.981 | 2.430 | 18.162 | 17.510 | 18.770 | 45 | 37 | 18 | 0 | 248 | 5 | 45 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p50__sea_bg_0p10__sea_cfg_0p10 | 0.50 | 0.10 | 0.10 | 542.662 | 223.704 | 2.426 | 17.984 | 16.710 | 19.300 | 45 | 37 | 18 | 0 | 248 | 5 | 45 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p50__sea_bg_0p10__sea_cfg_0p10 | 0.50 | 0.10 | 0.10 | 544.363 | 218.277 | 2.494 | 32.918 | 28.640 | 35.680 | 45 | 36 | 16 | 0 | 240 | 6 | 44 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p50__sea_bg_0p10__sea_cfg_0p10 | 0.50 | 0.10 | 0.10 | 544.006 | 218.678 | 2.488 | 18.955 | 18.040 | 20.060 | 45 | 36 | 17 | 0 | 240 | 5 | 45 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p50__sea_bg_0p10__sea_cfg_0p10 | 0.50 | 0.10 | 0.10 | 543.652 | 223.979 | 2.427 | 18.718 | 17.650 | 19.510 | 45 | 37 | 18 | 0 | 248 | 5 | 45 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p50__sea_bg_0p10__sea_cfg_0p10 | 0.50 | 0.10 | 0.10 | 544.915 | 218.510 | 2.494 | 23.882 | 22.820 | 24.540 | 45 | 36 | 17 | 0 | 240 | 5 | 45 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p50__sea_bg_0p10__sea_cfg_0p20 | 0.50 | 0.10 | 0.20 | 537.977 | 210.501 | 2.556 | 16.415 | 14.510 | 17.930 | 45 | 36 | 21 | 0 | 240 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p50__sea_bg_0p10__sea_cfg_0p20 | 0.50 | 0.10 | 0.20 | 538.450 | 210.281 | 2.561 | 19.575 | 18.570 | 21.140 | 45 | 36 | 21 | 0 | 240 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p50__sea_bg_0p10__sea_cfg_0p20 | 0.50 | 0.10 | 0.20 | 539.052 | 215.564 | 2.501 | 16.462 | 14.940 | 17.930 | 45 | 37 | 21 | 0 | 248 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p50__sea_bg_0p10__sea_cfg_0p20 | 0.50 | 0.10 | 0.20 | 538.562 | 210.473 | 2.559 | 20.535 | 19.840 | 20.960 | 45 | 36 | 21 | 0 | 240 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p50__sea_bg_0p10__sea_cfg_0p20 | 0.50 | 0.10 | 0.20 | 539.330 | 215.518 | 2.502 | 18.194 | 17.610 | 18.660 | 45 | 37 | 21 | 0 | 248 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p50__sea_bg_0p10__sea_cfg_0p20 | 0.50 | 0.10 | 0.20 | 542.662 | 217.390 | 2.496 | 18.042 | 17.070 | 19.430 | 45 | 38 | 22 | 0 | 248 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p50__sea_bg_0p10__sea_cfg_0p20 | 0.50 | 0.10 | 0.20 | 544.363 | 217.307 | 2.505 | 33.123 | 30.130 | 36.120 | 45 | 35 | 21 | 0 | 248 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p50__sea_bg_0p10__sea_cfg_0p20 | 0.50 | 0.10 | 0.20 | 544.006 | 212.084 | 2.565 | 19.078 | 18.050 | 20.200 | 45 | 36 | 21 | 0 | 240 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p50__sea_bg_0p10__sea_cfg_0p20 | 0.50 | 0.10 | 0.20 | 543.652 | 217.583 | 2.499 | 19.041 | 18.450 | 19.840 | 45 | 37 | 21 | 0 | 248 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p50__sea_bg_0p10__sea_cfg_0p20 | 0.50 | 0.10 | 0.20 | 544.915 | 212.137 | 2.569 | 22.408 | 21.340 | 23.060 | 45 | 37 | 21 | 0 | 240 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p50__sea_bg_0p10__sea_cfg_0p50 | 0.50 | 0.10 | 0.50 | 537.977 | 177.858 | 3.025 | 14.688 | 12.650 | 16.100 | 45 | 39 | 15 | 0 | 200 | 36 | 14 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p50__sea_bg_0p10__sea_cfg_0p50 | 0.50 | 0.10 | 0.50 | 538.450 | 183.389 | 2.936 | 16.922 | 15.660 | 18.020 | 45 | 39 | 16 | 0 | 208 | 35 | 15 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p50__sea_bg_0p10__sea_cfg_0p50 | 0.50 | 0.10 | 0.50 | 539.052 | 183.439 | 2.939 | 15.210 | 14.190 | 16.110 | 45 | 39 | 16 | 0 | 208 | 35 | 15 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p50__sea_bg_0p10__sea_cfg_0p50 | 0.50 | 0.10 | 0.50 | 538.562 | 177.899 | 3.027 | 20.294 | 18.920 | 20.900 | 45 | 39 | 15 | 0 | 200 | 36 | 14 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p50__sea_bg_0p10__sea_cfg_0p50 | 0.50 | 0.10 | 0.50 | 539.330 | 183.416 | 2.940 | 17.690 | 16.730 | 18.180 | 45 | 39 | 16 | 0 | 208 | 35 | 15 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p50__sea_bg_0p10__sea_cfg_0p50 | 0.50 | 0.10 | 0.50 | 542.662 | 184.801 | 2.936 | 16.314 | 14.290 | 19.220 | 45 | 39 | 16 | 0 | 208 | 35 | 15 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p50__sea_bg_0p10__sea_cfg_0p50 | 0.50 | 0.10 | 0.50 | 544.363 | 179.151 | 3.039 | 27.607 | 25.910 | 29.380 | 45 | 39 | 15 | 0 | 200 | 36 | 14 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p50__sea_bg_0p10__sea_cfg_0p50 | 0.50 | 0.10 | 0.50 | 544.006 | 179.285 | 3.034 | 18.733 | 18.110 | 19.280 | 45 | 39 | 15 | 0 | 200 | 36 | 14 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p50__sea_bg_0p10__sea_cfg_0p50 | 0.50 | 0.10 | 0.50 | 543.652 | 184.933 | 2.940 | 17.795 | 16.980 | 18.460 | 45 | 39 | 16 | 0 | 208 | 35 | 15 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p50__sea_bg_0p10__sea_cfg_0p50 | 0.50 | 0.10 | 0.50 | 544.915 | 179.247 | 3.040 | 22.889 | 22.250 | 23.280 | 45 | 39 | 15 | 0 | 200 | 36 | 14 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p50__sea_bg_0p20__sea_cfg_0p05 | 0.50 | 0.20 | 0.05 | 537.977 | 219.090 | 2.456 | 15.582 | 13.690 | 16.770 | 45 | 35 | 15 | 0 | 240 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p50__sea_bg_0p20__sea_cfg_0p05 | 0.50 | 0.20 | 0.05 | 538.450 | 218.766 | 2.461 | 19.977 | 18.770 | 21.640 | 45 | 35 | 15 | 0 | 240 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p50__sea_bg_0p20__sea_cfg_0p05 | 0.50 | 0.20 | 0.05 | 539.052 | 229.431 | 2.350 | 16.490 | 15.290 | 17.790 | 45 | 34 | 16 | 0 | 256 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p50__sea_bg_0p20__sea_cfg_0p05 | 0.50 | 0.20 | 0.05 | 538.562 | 219.126 | 2.458 | 22.051 | 20.950 | 22.520 | 45 | 35 | 15 | 0 | 240 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p50__sea_bg_0p20__sea_cfg_0p05 | 0.50 | 0.20 | 0.05 | 539.330 | 229.284 | 2.352 | 18.471 | 17.800 | 18.930 | 45 | 34 | 16 | 0 | 256 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p50__sea_bg_0p20__sea_cfg_0p05 | 0.50 | 0.20 | 0.05 | 542.662 | 230.959 | 2.350 | 18.148 | 16.760 | 19.420 | 45 | 34 | 16 | 0 | 256 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p50__sea_bg_0p20__sea_cfg_0p05 | 0.50 | 0.20 | 0.05 | 544.363 | 220.555 | 2.468 | 33.678 | 28.840 | 36.450 | 45 | 35 | 15 | 0 | 240 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p50__sea_bg_0p20__sea_cfg_0p05 | 0.50 | 0.20 | 0.05 | 544.006 | 220.802 | 2.464 | 18.941 | 18.030 | 20.040 | 45 | 35 | 15 | 0 | 240 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p50__sea_bg_0p20__sea_cfg_0p05 | 0.50 | 0.20 | 0.05 | 543.652 | 231.484 | 2.349 | 18.963 | 17.800 | 19.810 | 45 | 34 | 16 | 0 | 256 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p50__sea_bg_0p20__sea_cfg_0p05 | 0.50 | 0.20 | 0.05 | 544.915 | 220.533 | 2.471 | 24.487 | 23.160 | 25.250 | 45 | 35 | 15 | 0 | 240 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p50__sea_bg_0p20__sea_cfg_0p10 | 0.50 | 0.20 | 0.10 | 537.977 | 216.090 | 2.490 | 15.669 | 13.710 | 16.880 | 45 | 35 | 16 | 0 | 240 | 6 | 44 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p50__sea_bg_0p20__sea_cfg_0p10 | 0.50 | 0.20 | 0.10 | 538.450 | 216.676 | 2.485 | 19.899 | 18.700 | 21.540 | 45 | 36 | 17 | 0 | 240 | 5 | 45 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p50__sea_bg_0p20__sea_cfg_0p10 | 0.50 | 0.20 | 0.10 | 539.052 | 222.074 | 2.427 | 16.307 | 14.920 | 17.620 | 45 | 37 | 18 | 0 | 248 | 5 | 45 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p50__sea_bg_0p20__sea_cfg_0p10 | 0.50 | 0.20 | 0.10 | 538.562 | 216.587 | 2.487 | 21.885 | 20.950 | 22.320 | 45 | 35 | 16 | 0 | 240 | 6 | 44 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p50__sea_bg_0p20__sea_cfg_0p10 | 0.50 | 0.20 | 0.10 | 539.330 | 222.033 | 2.429 | 18.162 | 17.510 | 18.770 | 45 | 37 | 18 | 0 | 248 | 5 | 45 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p50__sea_bg_0p20__sea_cfg_0p10 | 0.50 | 0.20 | 0.10 | 542.662 | 223.761 | 2.425 | 17.984 | 16.710 | 19.300 | 45 | 37 | 18 | 0 | 248 | 5 | 45 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p50__sea_bg_0p20__sea_cfg_0p10 | 0.50 | 0.20 | 0.10 | 544.363 | 218.142 | 2.495 | 32.918 | 28.640 | 35.680 | 45 | 36 | 16 | 0 | 240 | 6 | 44 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p50__sea_bg_0p20__sea_cfg_0p10 | 0.50 | 0.20 | 0.10 | 544.006 | 218.604 | 2.489 | 18.955 | 18.040 | 20.060 | 45 | 36 | 17 | 0 | 240 | 5 | 45 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p50__sea_bg_0p20__sea_cfg_0p10 | 0.50 | 0.20 | 0.10 | 543.652 | 224.148 | 2.425 | 18.718 | 17.650 | 19.510 | 45 | 37 | 18 | 0 | 248 | 5 | 45 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p50__sea_bg_0p20__sea_cfg_0p10 | 0.50 | 0.20 | 0.10 | 544.915 | 218.677 | 2.492 | 23.882 | 22.820 | 24.540 | 45 | 36 | 17 | 0 | 240 | 5 | 45 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p50__sea_bg_0p20__sea_cfg_0p20 | 0.50 | 0.20 | 0.20 | 537.977 | 210.194 | 2.559 | 16.415 | 14.510 | 17.930 | 45 | 36 | 21 | 0 | 240 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p50__sea_bg_0p20__sea_cfg_0p20 | 0.50 | 0.20 | 0.20 | 538.450 | 210.408 | 2.559 | 19.575 | 18.570 | 21.140 | 45 | 36 | 21 | 0 | 240 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p50__sea_bg_0p20__sea_cfg_0p20 | 0.50 | 0.20 | 0.20 | 539.052 | 215.813 | 2.498 | 16.462 | 14.940 | 17.930 | 45 | 37 | 21 | 0 | 248 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p50__sea_bg_0p20__sea_cfg_0p20 | 0.50 | 0.20 | 0.20 | 538.562 | 210.415 | 2.560 | 20.535 | 19.840 | 20.960 | 45 | 36 | 21 | 0 | 240 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p50__sea_bg_0p20__sea_cfg_0p20 | 0.50 | 0.20 | 0.20 | 539.330 | 215.659 | 2.501 | 18.194 | 17.610 | 18.660 | 45 | 37 | 21 | 0 | 248 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p50__sea_bg_0p20__sea_cfg_0p20 | 0.50 | 0.20 | 0.20 | 542.662 | 217.479 | 2.495 | 18.042 | 17.070 | 19.430 | 45 | 38 | 22 | 0 | 248 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p50__sea_bg_0p20__sea_cfg_0p20 | 0.50 | 0.20 | 0.20 | 544.363 | 217.386 | 2.504 | 33.123 | 30.130 | 36.120 | 45 | 35 | 21 | 0 | 248 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p50__sea_bg_0p20__sea_cfg_0p20 | 0.50 | 0.20 | 0.20 | 544.006 | 212.278 | 2.563 | 19.078 | 18.050 | 20.200 | 45 | 36 | 21 | 0 | 240 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p50__sea_bg_0p20__sea_cfg_0p20 | 0.50 | 0.20 | 0.20 | 543.652 | 217.653 | 2.498 | 19.041 | 18.450 | 19.840 | 45 | 37 | 21 | 0 | 248 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p50__sea_bg_0p20__sea_cfg_0p20 | 0.50 | 0.20 | 0.20 | 544.915 | 212.276 | 2.567 | 22.408 | 21.340 | 23.060 | 45 | 37 | 21 | 0 | 240 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p50__sea_bg_0p20__sea_cfg_0p50 | 0.50 | 0.20 | 0.50 | 537.977 | 177.579 | 3.030 | 14.688 | 12.650 | 16.100 | 45 | 39 | 15 | 0 | 200 | 36 | 14 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p50__sea_bg_0p20__sea_cfg_0p50 | 0.50 | 0.20 | 0.50 | 538.450 | 183.484 | 2.935 | 16.922 | 15.660 | 18.020 | 45 | 39 | 16 | 0 | 208 | 35 | 15 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p50__sea_bg_0p20__sea_cfg_0p50 | 0.50 | 0.20 | 0.50 | 539.052 | 183.474 | 2.938 | 15.210 | 14.190 | 16.110 | 45 | 39 | 16 | 0 | 208 | 35 | 15 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p50__sea_bg_0p20__sea_cfg_0p50 | 0.50 | 0.20 | 0.50 | 538.562 | 177.788 | 3.029 | 20.294 | 18.920 | 20.900 | 45 | 39 | 15 | 0 | 200 | 36 | 14 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p50__sea_bg_0p20__sea_cfg_0p50 | 0.50 | 0.20 | 0.50 | 539.330 | 183.400 | 2.941 | 17.690 | 16.730 | 18.180 | 45 | 39 | 16 | 0 | 208 | 35 | 15 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p50__sea_bg_0p20__sea_cfg_0p50 | 0.50 | 0.20 | 0.50 | 542.662 | 184.847 | 2.936 | 16.314 | 14.290 | 19.220 | 45 | 39 | 16 | 0 | 208 | 35 | 15 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p50__sea_bg_0p20__sea_cfg_0p50 | 0.50 | 0.20 | 0.50 | 544.363 | 179.467 | 3.033 | 27.607 | 25.910 | 29.380 | 45 | 39 | 15 | 0 | 200 | 36 | 14 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p50__sea_bg_0p20__sea_cfg_0p50 | 0.50 | 0.20 | 0.50 | 544.006 | 179.353 | 3.033 | 18.733 | 18.110 | 19.280 | 45 | 39 | 15 | 0 | 200 | 36 | 14 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p50__sea_bg_0p20__sea_cfg_0p50 | 0.50 | 0.20 | 0.50 | 543.652 | 185.013 | 2.938 | 17.795 | 16.980 | 18.460 | 45 | 39 | 16 | 0 | 208 | 35 | 15 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p50__sea_bg_0p20__sea_cfg_0p50 | 0.50 | 0.20 | 0.50 | 544.915 | 179.416 | 3.037 | 22.889 | 22.250 | 23.280 | 45 | 39 | 15 | 0 | 200 | 36 | 14 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p50__sea_bg_0p50__sea_cfg_0p05 | 0.50 | 0.50 | 0.05 | 537.977 | 168.769 | 3.188 | 16.388 | 14.760 | 17.730 | 45 | 35 | 15 | 80 | 160 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p50__sea_bg_0p50__sea_cfg_0p05 | 0.50 | 0.50 | 0.05 | 538.450 | 168.800 | 3.190 | 17.839 | 16.500 | 18.810 | 45 | 35 | 15 | 80 | 160 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p50__sea_bg_0p50__sea_cfg_0p05 | 0.50 | 0.50 | 0.05 | 539.052 | 169.294 | 3.184 | 16.001 | 14.750 | 17.190 | 45 | 34 | 16 | 96 | 160 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p50__sea_bg_0p50__sea_cfg_0p05 | 0.50 | 0.50 | 0.05 | 538.562 | 168.856 | 3.189 | 20.892 | 20.300 | 21.300 | 45 | 35 | 15 | 80 | 160 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p50__sea_bg_0p50__sea_cfg_0p05 | 0.50 | 0.50 | 0.05 | 539.330 | 169.289 | 3.186 | 17.845 | 17.390 | 18.230 | 45 | 34 | 16 | 96 | 160 | 0 | 50 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p50__sea_bg_0p50__sea_cfg_0p05 | 0.50 | 0.50 | 0.05 | 542.662 | 170.510 | 3.183 | 17.276 | 15.040 | 19.620 | 45 | 34 | 16 | 96 | 160 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p50__sea_bg_0p50__sea_cfg_0p05 | 0.50 | 0.50 | 0.05 | 544.363 | 170.183 | 3.199 | 32.653 | 30.260 | 36.060 | 45 | 35 | 15 | 80 | 160 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p50__sea_bg_0p50__sea_cfg_0p05 | 0.50 | 0.50 | 0.05 | 544.006 | 170.240 | 3.196 | 19.951 | 19.140 | 20.700 | 45 | 35 | 15 | 80 | 160 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p50__sea_bg_0p50__sea_cfg_0p05 | 0.50 | 0.50 | 0.05 | 543.652 | 170.800 | 3.183 | 17.963 | 17.500 | 18.520 | 45 | 34 | 16 | 96 | 160 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p50__sea_bg_0p50__sea_cfg_0p05 | 0.50 | 0.50 | 0.05 | 544.915 | 170.235 | 3.201 | 24.352 | 23.730 | 24.800 | 45 | 35 | 15 | 80 | 160 | 0 | 50 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p50__sea_bg_0p50__sea_cfg_0p10 | 0.50 | 0.50 | 0.10 | 537.977 | 166.219 | 3.237 | 16.482 | 14.930 | 17.830 | 45 | 35 | 16 | 80 | 160 | 6 | 44 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p50__sea_bg_0p50__sea_cfg_0p10 | 0.50 | 0.50 | 0.10 | 538.450 | 166.787 | 3.228 | 17.568 | 16.140 | 18.660 | 45 | 36 | 17 | 80 | 160 | 5 | 45 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p50__sea_bg_0p50__sea_cfg_0p10 | 0.50 | 0.50 | 0.10 | 539.052 | 166.839 | 3.231 | 15.765 | 14.330 | 16.910 | 45 | 37 | 18 | 88 | 160 | 5 | 45 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p50__sea_bg_0p50__sea_cfg_0p10 | 0.50 | 0.50 | 0.10 | 538.562 | 166.378 | 3.237 | 20.706 | 20.140 | 21.130 | 45 | 35 | 16 | 80 | 160 | 6 | 44 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p50__sea_bg_0p50__sea_cfg_0p10 | 0.50 | 0.50 | 0.10 | 539.330 | 166.918 | 3.231 | 17.694 | 17.200 | 18.180 | 45 | 37 | 18 | 88 | 160 | 5 | 45 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p50__sea_bg_0p50__sea_cfg_0p10 | 0.50 | 0.50 | 0.10 | 542.662 | 168.121 | 3.228 | 17.113 | 14.780 | 19.980 | 45 | 37 | 18 | 88 | 160 | 5 | 45 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p50__sea_bg_0p50__sea_cfg_0p10 | 0.50 | 0.50 | 0.10 | 544.363 | 167.831 | 3.244 | 32.530 | 30.300 | 35.630 | 45 | 35 | 16 | 80 | 160 | 6 | 44 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p50__sea_bg_0p50__sea_cfg_0p10 | 0.50 | 0.50 | 0.10 | 544.006 | 168.040 | 3.237 | 19.151 | 18.700 | 19.760 | 45 | 36 | 17 | 80 | 160 | 5 | 45 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p50__sea_bg_0p50__sea_cfg_0p10 | 0.50 | 0.50 | 0.10 | 543.652 | 168.363 | 3.229 | 17.775 | 17.140 | 18.250 | 45 | 37 | 18 | 88 | 160 | 5 | 45 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p50__sea_bg_0p50__sea_cfg_0p10 | 0.50 | 0.50 | 0.10 | 544.915 | 168.163 | 3.240 | 22.479 | 21.180 | 23.540 | 45 | 36 | 17 | 80 | 160 | 5 | 45 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p50__sea_bg_0p50__sea_cfg_0p20 | 0.50 | 0.50 | 0.20 | 537.977 | 169.694 | 3.170 | 17.109 | 15.540 | 18.110 | 45 | 36 | 21 | 65 | 175 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p50__sea_bg_0p50__sea_cfg_0p20 | 0.50 | 0.50 | 0.20 | 538.450 | 169.746 | 3.172 | 17.488 | 16.250 | 18.960 | 45 | 36 | 21 | 65 | 175 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p50__sea_bg_0p50__sea_cfg_0p20 | 0.50 | 0.50 | 0.20 | 539.052 | 169.900 | 3.173 | 16.206 | 14.930 | 17.290 | 45 | 37 | 21 | 73 | 175 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p50__sea_bg_0p50__sea_cfg_0p20 | 0.50 | 0.50 | 0.20 | 538.562 | 170.507 | 3.159 | 18.301 | 17.710 | 18.800 | 45 | 36 | 21 | 64 | 176 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p50__sea_bg_0p50__sea_cfg_0p20 | 0.50 | 0.50 | 0.20 | 539.330 | 169.894 | 3.175 | 16.728 | 15.980 | 17.220 | 45 | 37 | 21 | 73 | 175 | 20 | 30 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p50__sea_bg_0p50__sea_cfg_0p20 | 0.50 | 0.50 | 0.20 | 542.662 | 171.329 | 3.167 | 16.400 | 14.940 | 18.470 | 45 | 37 | 21 | 73 | 175 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p50__sea_bg_0p50__sea_cfg_0p20 | 0.50 | 0.50 | 0.20 | 544.363 | 172.116 | 3.163 | 28.246 | 24.800 | 31.800 | 45 | 35 | 21 | 72 | 176 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p50__sea_bg_0p50__sea_cfg_0p20 | 0.50 | 0.50 | 0.20 | 544.006 | 170.933 | 3.183 | 18.659 | 17.910 | 19.480 | 45 | 36 | 21 | 65 | 175 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p50__sea_bg_0p50__sea_cfg_0p20 | 0.50 | 0.50 | 0.20 | 543.652 | 171.218 | 3.175 | 17.237 | 16.360 | 17.670 | 45 | 37 | 21 | 65 | 175 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p50__sea_bg_0p50__sea_cfg_0p20 | 0.50 | 0.50 | 0.20 | 544.915 | 171.270 | 3.182 | 21.083 | 20.260 | 21.630 | 45 | 37 | 21 | 65 | 175 | 20 | 30 | shard_gpu1_p005_009 |
| vbench10_001 | sea_ts_0p50__sea_bg_0p50__sea_cfg_0p50 | 0.50 | 0.50 | 0.50 | 537.977 | 152.728 | 3.522 | 16.406 | 14.980 | 17.830 | 45 | 39 | 15 | 40 | 160 | 36 | 14 | shard_gpu0_p000_004 |
| vbench10_002 | sea_ts_0p50__sea_bg_0p50__sea_cfg_0p50 | 0.50 | 0.50 | 0.50 | 538.450 | 152.645 | 3.527 | 17.831 | 16.690 | 18.890 | 45 | 39 | 15 | 40 | 160 | 36 | 14 | shard_gpu0_p000_004 |
| vbench10_003 | sea_ts_0p50__sea_bg_0p50__sea_cfg_0p50 | 0.50 | 0.50 | 0.50 | 539.052 | 153.277 | 3.517 | 16.027 | 14.770 | 17.320 | 45 | 39 | 16 | 48 | 160 | 35 | 15 | shard_gpu0_p000_004 |
| vbench10_004 | sea_ts_0p50__sea_bg_0p50__sea_cfg_0p50 | 0.50 | 0.50 | 0.50 | 538.562 | 152.731 | 3.526 | 20.816 | 20.280 | 21.240 | 45 | 39 | 15 | 40 | 160 | 36 | 14 | shard_gpu0_p000_004 |
| vbench10_005 | sea_ts_0p50__sea_bg_0p50__sea_cfg_0p50 | 0.50 | 0.50 | 0.50 | 539.330 | 153.251 | 3.519 | 17.860 | 17.410 | 18.280 | 45 | 39 | 16 | 48 | 160 | 35 | 15 | shard_gpu0_p000_004 |
| vbench10_006 | sea_ts_0p50__sea_bg_0p50__sea_cfg_0p50 | 0.50 | 0.50 | 0.50 | 542.662 | 154.424 | 3.514 | 17.245 | 14.930 | 19.650 | 45 | 39 | 16 | 48 | 160 | 35 | 15 | shard_gpu1_p005_009 |
| vbench10_007 | sea_ts_0p50__sea_bg_0p50__sea_cfg_0p50 | 0.50 | 0.50 | 0.50 | 544.363 | 153.966 | 3.536 | 32.565 | 30.140 | 36.040 | 45 | 39 | 15 | 40 | 160 | 36 | 14 | shard_gpu1_p005_009 |
| vbench10_008 | sea_ts_0p50__sea_bg_0p50__sea_cfg_0p50 | 0.50 | 0.50 | 0.50 | 544.006 | 154.050 | 3.531 | 19.904 | 19.150 | 20.660 | 45 | 39 | 15 | 40 | 160 | 36 | 14 | shard_gpu1_p005_009 |
| vbench10_009 | sea_ts_0p50__sea_bg_0p50__sea_cfg_0p50 | 0.50 | 0.50 | 0.50 | 543.652 | 154.897 | 3.510 | 17.684 | 16.820 | 18.500 | 45 | 39 | 16 | 48 | 160 | 35 | 15 | shard_gpu1_p005_009 |
| vbench10_010 | sea_ts_0p50__sea_bg_0p50__sea_cfg_0p50 | 0.50 | 0.50 | 0.50 | 544.915 | 153.956 | 3.539 | 24.240 | 23.530 | 24.690 | 45 | 39 | 15 | 40 | 160 | 36 | 14 | shard_gpu1_p005_009 |


## 3. 10-Prompt Aggregate Table

One row per threshold combination, aggregated across all 10 prompts.

| candidate | ts_th | bg_th | cfg_th | n | base_total_s | cand_total_s | overall_speedup | mean_psnr | min_psnr | t_reuse | t_recomp | bg_reuse | bg_recomp | cfg_reuse | cfg_recomp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sea_ts_0p05__sea_bg_0p05__sea_cfg_0p05 | 0.05 | 0.05 | 0.05 | 10 | 5412.969 | 5851.044 | 0.925 |  |  | 0 | 500 | 0 | 8000 | 0 | 500 |
| sea_ts_0p05__sea_bg_0p05__sea_cfg_0p10 | 0.05 | 0.05 | 0.10 | 10 | 5412.969 | 5535.921 | 0.978 | 37.499 | 29.110 | 0 | 500 | 0 | 7552 | 56 | 444 |
| sea_ts_0p05__sea_bg_0p05__sea_cfg_0p20 | 0.05 | 0.05 | 0.20 | 10 | 5412.969 | 4723.386 | 1.146 | 26.538 | 16.350 | 0 | 500 | 0 | 6400 | 200 | 300 |
| sea_ts_0p05__sea_bg_0p05__sea_cfg_0p50 | 0.05 | 0.05 | 0.50 | 10 | 5412.969 | 3843.214 | 1.408 | 20.640 | 13.020 | 0 | 500 | 0 | 5152 | 356 | 144 |
| sea_ts_0p05__sea_bg_0p10__sea_cfg_0p05 | 0.05 | 0.10 | 0.05 | 10 | 5412.969 | 5390.903 | 1.004 | 36.079 | 28.360 | 0 | 500 | 735 | 7265 | 0 | 500 |
| sea_ts_0p05__sea_bg_0p10__sea_cfg_0p10 | 0.05 | 0.10 | 0.10 | 10 | 5412.969 | 5309.338 | 1.020 | 36.126 | 28.270 | 0 | 500 | 364 | 7188 | 56 | 444 |
| sea_ts_0p05__sea_bg_0p10__sea_cfg_0p20 | 0.05 | 0.10 | 0.20 | 10 | 5412.969 | 4496.360 | 1.204 | 26.382 | 16.460 | 0 | 500 | 364 | 6036 | 200 | 300 |
| sea_ts_0p05__sea_bg_0p10__sea_cfg_0p50 | 0.05 | 0.10 | 0.50 | 10 | 5412.969 | 3616.984 | 1.497 | 20.776 | 13.100 | 0 | 500 | 364 | 4788 | 356 | 144 |
| sea_ts_0p05__sea_bg_0p20__sea_cfg_0p05 | 0.05 | 0.20 | 0.05 | 10 | 5412.969 | 3937.472 | 1.375 | 25.596 | 17.480 | 0 | 500 | 3049 | 4951 | 0 | 500 |
| sea_ts_0p05__sea_bg_0p20__sea_cfg_0p10 | 0.05 | 0.20 | 0.10 | 10 | 5412.969 | 3811.096 | 1.420 | 25.336 | 17.390 | 0 | 500 | 2720 | 4808 | 59 | 441 |
| sea_ts_0p05__sea_bg_0p20__sea_cfg_0p20 | 0.05 | 0.20 | 0.20 | 10 | 5412.969 | 3751.684 | 1.443 | 25.111 | 17.120 | 0 | 500 | 1548 | 4852 | 200 | 300 |
| sea_ts_0p05__sea_bg_0p20__sea_cfg_0p50 | 0.05 | 0.20 | 0.50 | 10 | 5412.969 | 2879.524 | 1.880 | 20.707 | 13.620 | 0 | 500 | 1531 | 3621 | 356 | 144 |
| sea_ts_0p05__sea_bg_0p50__sea_cfg_0p05 | 0.05 | 0.50 | 0.05 | 10 | 5412.969 | 2351.633 | 2.302 | 20.453 | 13.550 | 0 | 500 | 5567 | 2433 | 0 | 500 |
| sea_ts_0p05__sea_bg_0p50__sea_cfg_0p10 | 0.05 | 0.50 | 0.10 | 10 | 5412.969 | 2306.900 | 2.346 | 20.216 | 13.560 | 0 | 500 | 5153 | 2415 | 54 | 446 |
| sea_ts_0p05__sea_bg_0p50__sea_cfg_0p20 | 0.05 | 0.50 | 0.20 | 10 | 5412.969 | 2180.898 | 2.482 | 20.088 | 14.470 | 0 | 500 | 4043 | 2357 | 200 | 300 |
| sea_ts_0p05__sea_bg_0p50__sea_cfg_0p50 | 0.05 | 0.50 | 0.50 | 10 | 5412.969 | 1878.563 | 2.881 | 19.071 | 12.920 | 0 | 500 | 3133 | 2027 | 355 | 145 |
| sea_ts_0p10__sea_bg_0p05__sea_cfg_0p05 | 0.10 | 0.05 | 0.05 | 10 | 5412.969 | 5263.334 | 1.028 | 35.897 | 27.970 | 56 | 444 | 0 | 7104 | 0 | 500 |
| sea_ts_0p10__sea_bg_0p05__sea_cfg_0p10 | 0.10 | 0.05 | 0.10 | 10 | 5412.969 | 5243.182 | 1.032 | 36.113 | 28.450 | 56 | 444 | 0 | 7104 | 56 | 444 |
| sea_ts_0p10__sea_bg_0p05__sea_cfg_0p20 | 0.10 | 0.05 | 0.20 | 10 | 5412.969 | 4432.372 | 1.221 | 26.387 | 16.470 | 56 | 460 | 0 | 5952 | 200 | 300 |
| sea_ts_0p10__sea_bg_0p05__sea_cfg_0p50 | 0.10 | 0.05 | 0.50 | 10 | 5412.969 | 3545.593 | 1.527 | 20.770 | 13.080 | 57 | 449 | 0 | 4696 | 356 | 144 |
| sea_ts_0p10__sea_bg_0p10__sea_cfg_0p05 | 0.10 | 0.10 | 0.05 | 10 | 5412.969 | 5268.050 | 1.028 | 35.897 | 27.970 | 56 | 444 | 0 | 7104 | 0 | 500 |
| sea_ts_0p10__sea_bg_0p10__sea_cfg_0p10 | 0.10 | 0.10 | 0.10 | 10 | 5412.969 | 5244.811 | 1.032 | 36.113 | 28.450 | 56 | 444 | 0 | 7104 | 56 | 444 |
| sea_ts_0p10__sea_bg_0p10__sea_cfg_0p20 | 0.10 | 0.10 | 0.20 | 10 | 5412.969 | 4433.643 | 1.221 | 26.387 | 16.470 | 56 | 460 | 0 | 5952 | 200 | 300 |
| sea_ts_0p10__sea_bg_0p10__sea_cfg_0p50 | 0.10 | 0.10 | 0.50 | 10 | 5412.969 | 3545.900 | 1.527 | 20.770 | 13.080 | 57 | 449 | 0 | 4696 | 356 | 144 |
| sea_ts_0p10__sea_bg_0p20__sea_cfg_0p05 | 0.10 | 0.20 | 0.05 | 10 | 5412.969 | 3735.075 | 1.449 | 25.591 | 17.510 | 56 | 444 | 2439 | 4665 | 0 | 500 |
| sea_ts_0p10__sea_bg_0p20__sea_cfg_0p10 | 0.10 | 0.20 | 0.10 | 10 | 5412.969 | 3709.662 | 1.459 | 25.585 | 17.500 | 56 | 444 | 2439 | 4665 | 56 | 444 |
| sea_ts_0p10__sea_bg_0p20__sea_cfg_0p20 | 0.10 | 0.20 | 0.20 | 10 | 5412.969 | 3648.121 | 1.484 | 24.698 | 17.440 | 59 | 460 | 1219 | 4709 | 200 | 300 |
| sea_ts_0p10__sea_bg_0p20__sea_cfg_0p50 | 0.10 | 0.20 | 0.50 | 10 | 5412.969 | 2778.426 | 1.948 | 20.737 | 13.680 | 59 | 447 | 1202 | 3478 | 356 | 144 |
| sea_ts_0p10__sea_bg_0p50__sea_cfg_0p05 | 0.10 | 0.50 | 0.05 | 10 | 5412.969 | 2305.333 | 2.348 | 20.489 | 13.550 | 59 | 441 | 4659 | 2397 | 0 | 500 |
| sea_ts_0p10__sea_bg_0p50__sea_cfg_0p10 | 0.10 | 0.50 | 0.10 | 10 | 5412.969 | 2280.251 | 2.374 | 20.492 | 13.560 | 59 | 441 | 4659 | 2397 | 59 | 441 |
| sea_ts_0p10__sea_bg_0p50__sea_cfg_0p20 | 0.10 | 0.50 | 0.20 | 10 | 5412.969 | 2157.718 | 2.509 | 20.394 | 14.670 | 59 | 460 | 3589 | 2339 | 200 | 300 |
| sea_ts_0p10__sea_bg_0p50__sea_cfg_0p50 | 0.10 | 0.50 | 0.50 | 10 | 5412.969 | 1857.798 | 2.914 | 19.379 | 13.060 | 59 | 448 | 2657 | 2015 | 357 | 143 |
| sea_ts_0p20__sea_bg_0p05__sea_cfg_0p05 | 0.20 | 0.05 | 0.05 | 10 | 5412.969 | 3761.150 | 1.439 | 25.958 | 17.530 | 200 | 300 | 0 | 4800 | 0 | 500 |
| sea_ts_0p20__sea_bg_0p05__sea_cfg_0p10 | 0.20 | 0.05 | 0.10 | 10 | 5412.969 | 3686.982 | 1.468 | 25.631 | 17.420 | 215 | 324 | 0 | 4720 | 59 | 441 |
| sea_ts_0p20__sea_bg_0p05__sea_cfg_0p20 | 0.20 | 0.05 | 0.20 | 10 | 5412.969 | 3574.023 | 1.515 | 25.759 | 17.290 | 220 | 300 | 0 | 4640 | 200 | 300 |
| sea_ts_0p20__sea_bg_0p05__sea_cfg_0p50 | 0.20 | 0.05 | 0.50 | 10 | 5412.969 | 2798.645 | 1.934 | 20.704 | 13.580 | 200 | 340 | 0 | 3552 | 356 | 144 |
| sea_ts_0p20__sea_bg_0p10__sea_cfg_0p05 | 0.20 | 0.10 | 0.05 | 10 | 5412.969 | 3763.533 | 1.438 | 25.958 | 17.530 | 200 | 300 | 0 | 4800 | 0 | 500 |
| sea_ts_0p20__sea_bg_0p10__sea_cfg_0p10 | 0.20 | 0.10 | 0.10 | 10 | 5412.969 | 3686.793 | 1.468 | 25.631 | 17.420 | 215 | 324 | 0 | 4720 | 59 | 441 |
| sea_ts_0p20__sea_bg_0p10__sea_cfg_0p20 | 0.20 | 0.10 | 0.20 | 10 | 5412.969 | 3575.569 | 1.514 | 25.759 | 17.290 | 220 | 300 | 0 | 4640 | 200 | 300 |
| sea_ts_0p20__sea_bg_0p10__sea_cfg_0p50 | 0.20 | 0.10 | 0.50 | 10 | 5412.969 | 2798.394 | 1.934 | 20.704 | 13.580 | 200 | 340 | 0 | 3552 | 356 | 144 |
| sea_ts_0p20__sea_bg_0p20__sea_cfg_0p05 | 0.20 | 0.20 | 0.05 | 10 | 5412.969 | 3735.712 | 1.449 | 25.651 | 17.510 | 200 | 300 | 45 | 4755 | 0 | 500 |
| sea_ts_0p20__sea_bg_0p20__sea_cfg_0p10 | 0.20 | 0.20 | 0.10 | 10 | 5412.969 | 3659.931 | 1.479 | 25.341 | 17.380 | 215 | 324 | 45 | 4675 | 59 | 441 |
| sea_ts_0p20__sea_bg_0p20__sea_cfg_0p20 | 0.20 | 0.20 | 0.20 | 10 | 5412.969 | 3561.229 | 1.520 | 25.509 | 17.120 | 220 | 300 | 23 | 4617 | 200 | 300 |
| sea_ts_0p20__sea_bg_0p20__sea_cfg_0p50 | 0.20 | 0.20 | 0.50 | 10 | 5412.969 | 2786.174 | 1.943 | 20.725 | 13.610 | 200 | 340 | 22 | 3530 | 356 | 144 |
| sea_ts_0p20__sea_bg_0p50__sea_cfg_0p05 | 0.20 | 0.50 | 0.05 | 10 | 5412.969 | 2177.494 | 2.486 | 20.453 | 14.000 | 200 | 300 | 2525 | 2275 | 0 | 500 |
| sea_ts_0p20__sea_bg_0p50__sea_cfg_0p10 | 0.20 | 0.50 | 0.10 | 10 | 5412.969 | 2135.205 | 2.535 | 20.146 | 14.050 | 215 | 324 | 2469 | 2251 | 59 | 441 |
| sea_ts_0p20__sea_bg_0p50__sea_cfg_0p20 | 0.20 | 0.50 | 0.20 | 10 | 5412.969 | 2037.236 | 2.657 | 20.416 | 13.750 | 220 | 300 | 2445 | 2195 | 200 | 300 |
| sea_ts_0p20__sea_bg_0p50__sea_cfg_0p50 | 0.20 | 0.50 | 0.50 | 10 | 5412.969 | 1784.409 | 3.033 | 19.658 | 14.490 | 200 | 340 | 1613 | 1939 | 356 | 144 |
| sea_ts_0p50__sea_bg_0p05__sea_cfg_0p05 | 0.50 | 0.05 | 0.05 | 10 | 5412.969 | 2239.602 | 2.417 | 20.679 | 13.690 | 346 | 154 | 0 | 2464 | 0 | 500 |
| sea_ts_0p50__sea_bg_0p05__sea_cfg_0p10 | 0.50 | 0.05 | 0.10 | 10 | 5412.969 | 2196.217 | 2.465 | 20.438 | 13.710 | 362 | 171 | 0 | 2432 | 53 | 447 |
| sea_ts_0p50__sea_bg_0p05__sea_cfg_0p20 | 0.50 | 0.05 | 0.20 | 10 | 5412.969 | 2139.565 | 2.530 | 20.287 | 14.510 | 365 | 211 | 0 | 2440 | 200 | 300 |
| sea_ts_0p50__sea_bg_0p05__sea_cfg_0p50 | 0.50 | 0.05 | 0.50 | 10 | 5412.969 | 1812.788 | 2.986 | 18.814 | 12.650 | 390 | 155 | 0 | 2040 | 355 | 145 |
| sea_ts_0p50__sea_bg_0p10__sea_cfg_0p05 | 0.50 | 0.10 | 0.05 | 10 | 5412.969 | 2238.925 | 2.418 | 20.679 | 13.690 | 346 | 154 | 0 | 2464 | 0 | 500 |
| sea_ts_0p50__sea_bg_0p10__sea_cfg_0p10 | 0.50 | 0.10 | 0.10 | 10 | 5412.969 | 2196.542 | 2.464 | 20.438 | 13.710 | 362 | 171 | 0 | 2432 | 53 | 447 |
| sea_ts_0p50__sea_bg_0p10__sea_cfg_0p20 | 0.50 | 0.10 | 0.20 | 10 | 5412.969 | 2138.838 | 2.531 | 20.287 | 14.510 | 365 | 211 | 0 | 2440 | 200 | 300 |
| sea_ts_0p50__sea_bg_0p10__sea_cfg_0p50 | 0.50 | 0.10 | 0.50 | 10 | 5412.969 | 1813.418 | 2.985 | 18.814 | 12.650 | 390 | 155 | 0 | 2040 | 355 | 145 |
| sea_ts_0p50__sea_bg_0p20__sea_cfg_0p05 | 0.50 | 0.20 | 0.05 | 10 | 5412.969 | 2240.030 | 2.416 | 20.679 | 13.690 | 346 | 154 | 0 | 2464 | 0 | 500 |
| sea_ts_0p50__sea_bg_0p20__sea_cfg_0p10 | 0.50 | 0.20 | 0.10 | 10 | 5412.969 | 2196.792 | 2.464 | 20.438 | 13.710 | 362 | 171 | 0 | 2432 | 53 | 447 |
| sea_ts_0p50__sea_bg_0p20__sea_cfg_0p20 | 0.50 | 0.20 | 0.20 | 10 | 5412.969 | 2139.561 | 2.530 | 20.287 | 14.510 | 365 | 211 | 0 | 2440 | 200 | 300 |
| sea_ts_0p50__sea_bg_0p20__sea_cfg_0p50 | 0.50 | 0.20 | 0.50 | 10 | 5412.969 | 1813.821 | 2.984 | 18.814 | 12.650 | 390 | 155 | 0 | 2040 | 355 | 145 |
| sea_ts_0p50__sea_bg_0p50__sea_cfg_0p05 | 0.50 | 0.50 | 0.05 | 10 | 5412.969 | 1696.976 | 3.190 | 20.116 | 14.750 | 346 | 154 | 864 | 1600 | 0 | 500 |
| sea_ts_0p50__sea_bg_0p50__sea_cfg_0p10 | 0.50 | 0.50 | 0.10 | 10 | 5412.969 | 1673.659 | 3.234 | 19.726 | 14.330 | 361 | 171 | 832 | 1600 | 53 | 447 |
| sea_ts_0p50__sea_bg_0p50__sea_cfg_0p20 | 0.50 | 0.50 | 0.20 | 10 | 5412.969 | 1706.607 | 3.172 | 18.746 | 14.930 | 364 | 210 | 680 | 1752 | 200 | 300 |
| sea_ts_0p50__sea_bg_0p50__sea_cfg_0p50 | 0.50 | 0.50 | 0.50 | 10 | 5412.969 | 1535.925 | 3.524 | 20.058 | 14.770 | 390 | 154 | 432 | 1600 | 356 | 144 |


## 4. Simple Analysis

- Fastest setting: `sea_ts_0p50__sea_bg_0p50__sea_cfg_0p50` with overall speedup `3.524` and mean PSNR `20.058`. This is an aggressive setting and quality is low compared with the highest-PSNR settings.

- Highest mean PSNR: `sea_ts_0p05__sea_bg_0p05__sea_cfg_0p10` with mean PSNR `37.499` and speedup `0.978`; it is slightly slower than baseline.

- Best quality while still faster than baseline: `sea_ts_0p05__sea_bg_0p10__sea_cfg_0p10` with speedup `1.020` and mean PSNR `36.126`.

- Among combinations with mean PSNR >= 30, the fastest is `sea_ts_0p10__sea_bg_0p05__sea_cfg_0p10` with speedup `1.032` and mean PSNR `36.113`.

- If the target is at least 2x speedup, the best mean-PSNR option is `sea_ts_0p50__sea_bg_0p05__sea_cfg_0p05` with speedup `2.417` and mean PSNR `20.679`.

- The grid shows the expected tradeoff: higher thresholds increase reuse and speed, but PSNR drops. CFG threshold has a strong effect: low CFG thresholds preserve quality, while `cfg=0.50` often drives high speed at visibly lower PSNR.
