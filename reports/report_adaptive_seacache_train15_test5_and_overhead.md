# Adaptive SeaCache Train15/Test5 And Overhead Report

Date: 2026-06-23

## 1. Experiment Environment, Configuration, And Parameters

### Environment

| Item | Value |
| --- | --- |
| Workspace | /hy-tmp/work/Wan2.2 |
| Conda environment | /hy-tmp/miniconda3/envs/Wan2.2 |
| Model checkpoint | /hy-tmp/models/Wan2.2-T2V-A14B |
| GPU | NVIDIA A100 80GB PCIe |
| CUDA/driver observed in project notes | CUDA 12.8, driver 570.211.01 |
| Prompt source | test_sets/openvid_100/prompts.jsonl |
| Baseline root | /hy-tmp/work/Wan2.2/experiment_results/openvid_100_seacache_trace_data |
| Adaptive predictor | /hy-tmp/wan22_adaptive_threshold_feature_ablation_hdim16_20260616/temporal_mean/best_model.pt |
| Feature set / hidden dim | temporal_mean / 16 |
| Video validation | 90 ffprobe files checked as 832x480, 45 frames |

### Shared Inference Parameters

| Parameter | Value |
| --- | --- |
| task | t2v-A14B |
| size | 832*480 |
| frame_num | 45 |
| sample_steps | 50 |
| sample_solver | dpm++ |
| base_seed | 42 |
| sample_shift | 12.0 |
| sample_guide_scale | (3.0, 4.0) |
| offload_model | true |
| convert_model_dtype | true |
| cache mode | timestep-only adaptive SeaCache; block_cache=None; cfg_cache=None |
| target PSNRs | 20, 25, 30 |
| quality metric | FFmpeg PSNR against matching no-cache baseline |
| speed metric | inference_compute_elapsed_seconds based speedup |

### Experiment Roots

| Experiment | Root | Rows | Status |
| --- | --- | --- | --- |
| train15/test5 adaptive inference | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521 | 60 candidate rows | completed, failed=0 |
| train5 overhead online-vs-replay | /hy-tmp/wan22_adaptive_seacache_overhead_train5_50step_45f_480p_20260619_143632 | 30 rows = 15 online + 15 replay | completed, failed=0 |

## 2. Complete Result Tables

The tables below include every candidate row with the key metrics needed for comparison. The raw CSV files remain the source of truth and include full prompt text and all archived paths.

### 2.1 Train15/Test5 Complete Candidate Table

| order | sample_id | source_id | split | target | elapsed_s | baseline_s | speedup | psnr_db | thr_min | thr_max | thr_mean | reuse | recompute | trace_rows | video_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | openvid_086 | openvidhd_part1_085 | train | 20 | 171.429 | 544.606 | 3.177 | 25.497 | 0.5781 | 0.6602 | 0.6423 | 74 | 26 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_20/openvidhd_part1_085.mp4 |
| 1 | openvid_086 | openvidhd_part1_085 | train | 25 | 190.625 | 544.606 | 2.857 | 25.432 | 0.4004 | 0.6211 | 0.5530 | 70 | 30 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_25/openvidhd_part1_085.mp4 |
| 1 | openvid_086 | openvidhd_part1_085 | train | 30 | 229.979 | 544.606 | 2.368 | 23.510 | 0.2598 | 0.4766 | 0.3927 | 62 | 38 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_30/openvidhd_part1_085.mp4 |
| 2 | openvid_087 | openvidhd_part1_086 | train | 20 | 170.686 | 545.038 | 3.193 | 18.500 | 0.5234 | 0.5898 | 0.5736 | 74 | 26 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_20/openvidhd_part1_086.mp4 |
| 2 | openvid_087 | openvidhd_part1_086 | train | 25 | 220.401 | 545.038 | 2.473 | 21.696 | 0.3398 | 0.4180 | 0.3895 | 64 | 36 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_25/openvidhd_part1_086.mp4 |
| 2 | openvid_087 | openvidhd_part1_086 | train | 30 | 289.438 | 545.038 | 1.883 | 26.736 | 0.1934 | 0.2617 | 0.2349 | 50 | 50 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_30/openvidhd_part1_086.mp4 |
| 3 | openvid_060 | openvidhd_part1_059 | train | 20 | 171.397 | 543.412 | 3.170 | 21.045 | 0.5781 | 0.6875 | 0.6203 | 74 | 26 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_20/openvidhd_part1_059.mp4 |
| 3 | openvid_060 | openvidhd_part1_059 | train | 25 | 240.215 | 543.412 | 2.262 | 20.972 | 0.2891 | 0.4570 | 0.3484 | 60 | 40 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_25/openvidhd_part1_059.mp4 |
| 3 | openvid_060 | openvidhd_part1_059 | train | 30 | 328.739 | 543.412 | 1.653 | 34.824 | 0.1621 | 0.3203 | 0.2058 | 42 | 58 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_30/openvidhd_part1_059.mp4 |
| 4 | openvid_058 | openvidhd_part1_057 | train | 20 | 161.345 | 542.941 | 3.365 | 29.135 | 0.5781 | 0.7383 | 0.6570 | 76 | 24 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_20/openvidhd_part1_057.mp4 |
| 4 | openvid_058 | openvidhd_part1_057 | train | 25 | 200.509 | 542.941 | 2.708 | 31.613 | 0.4004 | 0.7070 | 0.5686 | 68 | 32 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_25/openvidhd_part1_057.mp4 |
| 4 | openvid_058 | openvidhd_part1_057 | train | 30 | 229.993 | 542.941 | 2.361 | 33.890 | 0.2598 | 0.6406 | 0.4149 | 62 | 38 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_30/openvidhd_part1_057.mp4 |
| 5 | openvid_017 | openvidhd_part1_016 | train | 20 | 169.852 | 535.343 | 3.152 | 20.344 | 0.5781 | 0.7188 | 0.6706 | 74 | 26 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_20/openvidhd_part1_016.mp4 |
| 5 | openvid_017 | openvidhd_part1_016 | train | 25 | 190.200 | 535.343 | 2.815 | 28.819 | 0.4004 | 0.6797 | 0.5804 | 70 | 30 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_25/openvidhd_part1_016.mp4 |
| 5 | openvid_017 | openvidhd_part1_016 | train | 30 | 239.418 | 535.343 | 2.236 | 33.833 | 0.2598 | 0.4922 | 0.3855 | 60 | 40 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_30/openvidhd_part1_016.mp4 |
| 6 | openvid_037 | openvidhd_part1_036 | train | 20 | 170.587 | 545.763 | 3.199 | 25.472 | 0.5781 | 0.5898 | 0.5830 | 74 | 26 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_20/openvidhd_part1_036.mp4 |
| 6 | openvid_037 | openvidhd_part1_036 | train | 25 | 199.949 | 545.763 | 2.730 | 30.389 | 0.4004 | 0.5703 | 0.4669 | 68 | 32 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_25/openvidhd_part1_036.mp4 |
| 6 | openvid_037 | openvidhd_part1_036 | train | 30 | 249.004 | 545.763 | 2.192 | 29.818 | 0.2598 | 0.5156 | 0.3282 | 58 | 42 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_30/openvidhd_part1_036.mp4 |
| 7 | openvid_094 | openvidhd_part1_093 | train | 20 | 180.384 | 545.019 | 3.021 | 16.296 | 0.4648 | 0.5781 | 0.5394 | 72 | 28 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_20/openvidhd_part1_093.mp4 |
| 7 | openvid_094 | openvidhd_part1_093 | train | 25 | 268.912 | 545.019 | 2.027 | 21.750 | 0.2266 | 0.4004 | 0.3052 | 54 | 46 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_25/openvidhd_part1_093.mp4 |
| 7 | openvid_094 | openvidhd_part1_093 | train | 30 | 337.808 | 545.019 | 1.613 | 30.857 | 0.1396 | 0.3320 | 0.1970 | 40 | 60 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_30/openvidhd_part1_093.mp4 |
| 8 | openvid_064 | openvidhd_part1_063 | train | 20 | 170.493 | 543.548 | 3.188 | 20.083 | 0.5781 | 0.7266 | 0.6089 | 74 | 26 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_20/openvidhd_part1_063.mp4 |
| 8 | openvid_064 | openvidhd_part1_063 | train | 25 | 219.455 | 543.548 | 2.477 | 20.636 | 0.3887 | 0.5234 | 0.4092 | 64 | 36 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_25/openvidhd_part1_063.mp4 |
| 8 | openvid_064 | openvidhd_part1_063 | train | 30 | 298.543 | 543.548 | 1.821 | 23.334 | 0.1787 | 0.2637 | 0.2254 | 48 | 52 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_30/openvidhd_part1_063.mp4 |
| 9 | openvid_096 | openvidhd_part1_095 | train | 20 | 170.884 | 544.604 | 3.187 | 15.416 | 0.5781 | 0.6250 | 0.6016 | 74 | 26 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_20/openvidhd_part1_095.mp4 |
| 9 | openvid_096 | openvidhd_part1_095 | train | 25 | 229.854 | 544.604 | 2.369 | 24.326 | 0.3320 | 0.4219 | 0.3892 | 62 | 38 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_25/openvidhd_part1_095.mp4 |
| 9 | openvid_096 | openvidhd_part1_095 | train | 30 | 308.557 | 544.604 | 1.765 | 27.455 | 0.1729 | 0.2637 | 0.2260 | 46 | 54 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_30/openvidhd_part1_095.mp4 |
| 10 | openvid_059 | openvidhd_part1_058 | train | 20 | 170.409 | 544.003 | 3.192 | 19.015 | 0.5586 | 0.6289 | 0.5845 | 74 | 26 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_20/openvidhd_part1_058.mp4 |
| 10 | openvid_059 | openvidhd_part1_058 | train | 25 | 237.674 | 544.003 | 2.289 | 19.479 | 0.2422 | 0.4004 | 0.3271 | 60 | 40 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_25/openvidhd_part1_058.mp4 |
| 10 | openvid_059 | openvidhd_part1_058 | train | 30 | 337.601 | 544.003 | 1.611 | 20.913 | 0.1309 | 0.2598 | 0.1944 | 40 | 60 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_30/openvidhd_part1_058.mp4 |
| 11 | openvid_028 | openvidhd_part1_027 | train | 20 | 180.068 | 544.808 | 3.026 | 17.629 | 0.4863 | 0.5820 | 0.5363 | 72 | 28 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_20/openvidhd_part1_027.mp4 |
| 11 | openvid_028 | openvidhd_part1_027 | train | 25 | 258.958 | 544.808 | 2.104 | 22.512 | 0.2266 | 0.4023 | 0.3077 | 56 | 44 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_25/openvidhd_part1_027.mp4 |
| 11 | openvid_028 | openvidhd_part1_027 | train | 30 | 328.034 | 544.808 | 1.661 | 34.115 | 0.1504 | 0.2598 | 0.1973 | 42 | 58 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_30/openvidhd_part1_027.mp4 |
| 12 | openvid_013 | openvidhd_part1_012 | train | 20 | 180.605 | 537.369 | 2.975 | 18.649 | 0.4219 | 0.5820 | 0.5332 | 72 | 28 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_20/openvidhd_part1_012.mp4 |
| 12 | openvid_013 | openvidhd_part1_012 | train | 25 | 259.148 | 537.369 | 2.074 | 20.674 | 0.2178 | 0.4023 | 0.2888 | 56 | 44 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_25/openvidhd_part1_012.mp4 |
| 12 | openvid_013 | openvidhd_part1_012 | train | 30 | 348.136 | 537.369 | 1.544 | 19.993 | 0.1338 | 0.2598 | 0.1821 | 38 | 62 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_30/openvidhd_part1_012.mp4 |
| 13 | openvid_021 | openvidhd_part1_020 | train | 20 | 160.535 | 542.703 | 3.381 | 24.818 | 0.5781 | 0.7344 | 0.6434 | 76 | 24 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_20/openvidhd_part1_020.mp4 |
| 13 | openvid_021 | openvidhd_part1_020 | train | 25 | 209.685 | 542.703 | 2.588 | 22.021 | 0.4004 | 0.6016 | 0.4518 | 66 | 34 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_25/openvidhd_part1_020.mp4 |
| 13 | openvid_021 | openvidhd_part1_020 | train | 30 | 258.976 | 542.703 | 2.096 | 27.849 | 0.2598 | 0.6016 | 0.3230 | 56 | 44 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_30/openvidhd_part1_020.mp4 |
| 14 | openvid_032 | openvidhd_part1_031 | train | 20 | 170.366 | 544.758 | 3.198 | 17.756 | 0.5000 | 0.6055 | 0.5821 | 74 | 26 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_20/openvidhd_part1_031.mp4 |
| 14 | openvid_032 | openvidhd_part1_031 | train | 25 | 239.138 | 544.758 | 2.278 | 17.671 | 0.1787 | 0.4375 | 0.3639 | 60 | 40 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_25/openvidhd_part1_031.mp4 |
| 14 | openvid_032 | openvidhd_part1_031 | train | 30 | 308.352 | 544.758 | 1.767 | 21.725 | 0.1157 | 0.2793 | 0.2321 | 46 | 54 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_30/openvidhd_part1_031.mp4 |
| 15 | openvid_038 | openvidhd_part1_037 | train | 20 | 180.576 | 546.036 | 3.024 | 17.563 | 0.4766 | 0.5781 | 0.5423 | 72 | 28 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_20/openvidhd_part1_037.mp4 |
| 15 | openvid_038 | openvidhd_part1_037 | train | 25 | 249.445 | 546.036 | 2.189 | 19.059 | 0.2500 | 0.4004 | 0.3105 | 58 | 42 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_25/openvidhd_part1_037.mp4 |
| 15 | openvid_038 | openvidhd_part1_037 | train | 30 | 337.905 | 546.036 | 1.616 | 23.295 | 0.1504 | 0.2812 | 0.1963 | 40 | 60 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_30/openvidhd_part1_037.mp4 |
| 16 | openvid_029 | openvidhd_part1_028 | test | 20 | 170.029 | 544.788 | 3.204 | 33.857 | 0.5781 | 0.6953 | 0.6580 | 74 | 26 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_20/openvidhd_part1_028.mp4 |
| 16 | openvid_029 | openvidhd_part1_028 | test | 25 | 189.578 | 544.788 | 2.874 | 31.125 | 0.4004 | 0.6641 | 0.5732 | 70 | 30 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_25/openvidhd_part1_028.mp4 |
| 16 | openvid_029 | openvidhd_part1_028 | test | 30 | 218.701 | 544.788 | 2.491 | 31.522 | 0.2598 | 0.5195 | 0.4019 | 64 | 36 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_30/openvidhd_part1_028.mp4 |
| 17 | openvid_031 | openvidhd_part1_030 | test | 20 | 169.659 | 544.506 | 3.209 | 20.822 | 0.5781 | 0.6758 | 0.6341 | 74 | 26 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_20/openvidhd_part1_030.mp4 |
| 17 | openvid_031 | openvidhd_part1_030 | test | 25 | 198.825 | 544.506 | 2.739 | 21.175 | 0.4004 | 0.5898 | 0.5042 | 68 | 32 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_25/openvidhd_part1_030.mp4 |
| 17 | openvid_031 | openvidhd_part1_030 | test | 30 | 267.811 | 544.506 | 2.033 | 25.756 | 0.1611 | 0.2969 | 0.2778 | 54 | 46 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_30/openvidhd_part1_030.mp4 |
| 18 | openvid_027 | openvidhd_part1_026 | test | 20 | 169.995 | 540.090 | 3.177 | 20.289 | 0.4883 | 0.5781 | 0.5492 | 74 | 26 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_20/openvidhd_part1_026.mp4 |
| 18 | openvid_027 | openvidhd_part1_026 | test | 25 | 237.427 | 540.090 | 2.275 | 21.177 | 0.3535 | 0.4004 | 0.3650 | 60 | 40 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_25/openvidhd_part1_026.mp4 |
| 18 | openvid_027 | openvidhd_part1_026 | test | 30 | 296.719 | 540.090 | 1.820 | 23.797 | 0.2324 | 0.2754 | 0.2417 | 48 | 52 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_30/openvidhd_part1_026.mp4 |
| 19 | openvid_093 | openvidhd_part1_092 | test | 20 | 169.823 | 544.770 | 3.208 | 19.355 | 0.5781 | 0.6719 | 0.6198 | 74 | 26 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_20/openvidhd_part1_092.mp4 |
| 19 | openvid_093 | openvidhd_part1_092 | test | 25 | 218.116 | 544.770 | 2.498 | 22.411 | 0.3398 | 0.4102 | 0.3871 | 64 | 36 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_25/openvidhd_part1_092.mp4 |
| 19 | openvid_093 | openvidhd_part1_092 | test | 30 | 328.011 | 544.770 | 1.661 | 25.288 | 0.1572 | 0.2617 | 0.2173 | 42 | 58 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_30/openvidhd_part1_092.mp4 |
| 20 | openvid_056 | openvidhd_part1_055 | test | 20 | 170.846 | 543.423 | 3.181 | 20.619 | 0.5781 | 0.6484 | 0.5974 | 74 | 26 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_20/openvidhd_part1_055.mp4 |
| 20 | openvid_056 | openvidhd_part1_055 | test | 25 | 209.899 | 543.423 | 2.589 | 25.208 | 0.4004 | 0.4766 | 0.4219 | 66 | 34 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_25/openvidhd_part1_055.mp4 |
| 20 | openvid_056 | openvidhd_part1_055 | test | 30 | 288.599 | 543.423 | 1.883 | 25.908 | 0.2256 | 0.2695 | 0.2462 | 50 | 50 | 100 | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/adaptive_seacache/target_30/openvidhd_part1_055.mp4 |

### 2.2 Overhead Complete Candidate Table

| order | sample_id | source_id | target | method | elapsed_s | baseline_s | speedup | psnr_db | thr_mean | reuse | recompute | predictor_total_s | predictor_calls | decision_mismatch | video_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | openvid_086 | openvidhd_part1_085 | 20 | online_adaptive | 171.573 | 544.606 | 3.174 | 25.497 | 0.6423 | 74 | 26 | 0.236853 | 100 |  | /hy-tmp/wan22_adaptive_seacache_overhead_train5_50step_45f_480p_20260619_143632/online_adaptive/target_20/openvidhd_part1_085.mp4 |
| 1 | openvid_086 | openvidhd_part1_085 | 20 | replay_threshold | 170.868 | 544.606 | 3.187 | 25.497 | 0.6423 | 74 | 26 |  | 0 | 0 | /hy-tmp/wan22_adaptive_seacache_overhead_train5_50step_45f_480p_20260619_143632/replay_threshold/target_20/openvidhd_part1_085.mp4 |
| 1 | openvid_086 | openvidhd_part1_085 | 25 | online_adaptive | 190.627 | 544.606 | 2.857 | 25.432 | 0.5530 | 70 | 30 | 0.175867 | 100 |  | /hy-tmp/wan22_adaptive_seacache_overhead_train5_50step_45f_480p_20260619_143632/online_adaptive/target_25/openvidhd_part1_085.mp4 |
| 1 | openvid_086 | openvidhd_part1_085 | 25 | replay_threshold | 190.636 | 544.606 | 2.857 | 25.432 | 0.5530 | 70 | 30 |  | 0 | 0 | /hy-tmp/wan22_adaptive_seacache_overhead_train5_50step_45f_480p_20260619_143632/replay_threshold/target_25/openvidhd_part1_085.mp4 |
| 1 | openvid_086 | openvidhd_part1_085 | 30 | online_adaptive | 230.027 | 544.606 | 2.368 | 23.510 | 0.3927 | 62 | 38 | 0.207156 | 100 |  | /hy-tmp/wan22_adaptive_seacache_overhead_train5_50step_45f_480p_20260619_143632/online_adaptive/target_30/openvidhd_part1_085.mp4 |
| 1 | openvid_086 | openvidhd_part1_085 | 30 | replay_threshold | 230.132 | 544.606 | 2.366 | 23.510 | 0.3927 | 62 | 38 |  | 0 | 0 | /hy-tmp/wan22_adaptive_seacache_overhead_train5_50step_45f_480p_20260619_143632/replay_threshold/target_30/openvidhd_part1_085.mp4 |
| 2 | openvid_087 | openvidhd_part1_086 | 20 | online_adaptive | 171.059 | 545.038 | 3.186 | 18.500 | 0.5736 | 74 | 26 | 0.178457 | 100 |  | /hy-tmp/wan22_adaptive_seacache_overhead_train5_50step_45f_480p_20260619_143632/online_adaptive/target_20/openvidhd_part1_086.mp4 |
| 2 | openvid_087 | openvidhd_part1_086 | 20 | replay_threshold | 170.916 | 545.038 | 3.189 | 18.500 | 0.5736 | 74 | 26 |  | 0 | 0 | /hy-tmp/wan22_adaptive_seacache_overhead_train5_50step_45f_480p_20260619_143632/replay_threshold/target_20/openvidhd_part1_086.mp4 |
| 2 | openvid_087 | openvidhd_part1_086 | 25 | online_adaptive | 220.155 | 545.038 | 2.476 | 21.696 | 0.3895 | 64 | 36 | 0.204076 | 100 |  | /hy-tmp/wan22_adaptive_seacache_overhead_train5_50step_45f_480p_20260619_143632/online_adaptive/target_25/openvidhd_part1_086.mp4 |
| 2 | openvid_087 | openvidhd_part1_086 | 25 | replay_threshold | 220.190 | 545.038 | 2.475 | 21.696 | 0.3895 | 64 | 36 |  | 0 | 0 | /hy-tmp/wan22_adaptive_seacache_overhead_train5_50step_45f_480p_20260619_143632/replay_threshold/target_25/openvidhd_part1_086.mp4 |
| 2 | openvid_087 | openvidhd_part1_086 | 30 | online_adaptive | 288.661 | 545.038 | 1.888 | 26.736 | 0.2349 | 50 | 50 | 0.197612 | 100 |  | /hy-tmp/wan22_adaptive_seacache_overhead_train5_50step_45f_480p_20260619_143632/online_adaptive/target_30/openvidhd_part1_086.mp4 |
| 2 | openvid_087 | openvidhd_part1_086 | 30 | replay_threshold | 288.828 | 545.038 | 1.887 | 26.736 | 0.2349 | 50 | 50 |  | 0 | 0 | /hy-tmp/wan22_adaptive_seacache_overhead_train5_50step_45f_480p_20260619_143632/replay_threshold/target_30/openvidhd_part1_086.mp4 |
| 3 | openvid_060 | openvidhd_part1_059 | 20 | online_adaptive | 171.092 | 543.412 | 3.176 | 21.045 | 0.6203 | 74 | 26 | 0.215884 | 100 |  | /hy-tmp/wan22_adaptive_seacache_overhead_train5_50step_45f_480p_20260619_143632/online_adaptive/target_20/openvidhd_part1_059.mp4 |
| 3 | openvid_060 | openvidhd_part1_059 | 20 | replay_threshold | 170.710 | 543.412 | 3.183 | 21.045 | 0.6203 | 74 | 26 |  | 0 | 0 | /hy-tmp/wan22_adaptive_seacache_overhead_train5_50step_45f_480p_20260619_143632/replay_threshold/target_20/openvidhd_part1_059.mp4 |
| 3 | openvid_060 | openvidhd_part1_059 | 25 | online_adaptive | 240.003 | 543.412 | 2.264 | 20.972 | 0.3484 | 60 | 40 | 0.187398 | 100 |  | /hy-tmp/wan22_adaptive_seacache_overhead_train5_50step_45f_480p_20260619_143632/online_adaptive/target_25/openvidhd_part1_059.mp4 |
| 3 | openvid_060 | openvidhd_part1_059 | 25 | replay_threshold | 239.667 | 543.412 | 2.267 | 20.972 | 0.3484 | 60 | 40 |  | 0 | 0 | /hy-tmp/wan22_adaptive_seacache_overhead_train5_50step_45f_480p_20260619_143632/replay_threshold/target_25/openvidhd_part1_059.mp4 |
| 3 | openvid_060 | openvidhd_part1_059 | 30 | online_adaptive | 328.533 | 543.412 | 1.654 | 34.824 | 0.2058 | 42 | 58 | 0.169179 | 100 |  | /hy-tmp/wan22_adaptive_seacache_overhead_train5_50step_45f_480p_20260619_143632/online_adaptive/target_30/openvidhd_part1_059.mp4 |
| 3 | openvid_060 | openvidhd_part1_059 | 30 | replay_threshold | 328.113 | 543.412 | 1.656 | 34.824 | 0.2058 | 42 | 58 |  | 0 | 0 | /hy-tmp/wan22_adaptive_seacache_overhead_train5_50step_45f_480p_20260619_143632/replay_threshold/target_30/openvidhd_part1_059.mp4 |
| 4 | openvid_058 | openvidhd_part1_057 | 20 | online_adaptive | 161.540 | 542.941 | 3.361 | 29.135 | 0.6570 | 76 | 24 | 0.185754 | 100 |  | /hy-tmp/wan22_adaptive_seacache_overhead_train5_50step_45f_480p_20260619_143632/online_adaptive/target_20/openvidhd_part1_057.mp4 |
| 4 | openvid_058 | openvidhd_part1_057 | 20 | replay_threshold | 160.937 | 542.941 | 3.374 | 29.135 | 0.6570 | 76 | 24 |  | 0 | 0 | /hy-tmp/wan22_adaptive_seacache_overhead_train5_50step_45f_480p_20260619_143632/replay_threshold/target_20/openvidhd_part1_057.mp4 |
| 4 | openvid_058 | openvidhd_part1_057 | 25 | online_adaptive | 200.718 | 542.941 | 2.705 | 31.613 | 0.5686 | 68 | 32 | 0.189711 | 100 |  | /hy-tmp/wan22_adaptive_seacache_overhead_train5_50step_45f_480p_20260619_143632/online_adaptive/target_25/openvidhd_part1_057.mp4 |
| 4 | openvid_058 | openvidhd_part1_057 | 25 | replay_threshold | 200.593 | 542.941 | 2.707 | 31.613 | 0.5686 | 68 | 32 |  | 0 | 0 | /hy-tmp/wan22_adaptive_seacache_overhead_train5_50step_45f_480p_20260619_143632/replay_threshold/target_25/openvidhd_part1_057.mp4 |
| 4 | openvid_058 | openvidhd_part1_057 | 30 | online_adaptive | 229.182 | 542.941 | 2.369 | 33.890 | 0.4149 | 62 | 38 | 0.194114 | 100 |  | /hy-tmp/wan22_adaptive_seacache_overhead_train5_50step_45f_480p_20260619_143632/online_adaptive/target_30/openvidhd_part1_057.mp4 |
| 4 | openvid_058 | openvidhd_part1_057 | 30 | replay_threshold | 229.159 | 542.941 | 2.369 | 33.890 | 0.4149 | 62 | 38 |  | 0 | 0 | /hy-tmp/wan22_adaptive_seacache_overhead_train5_50step_45f_480p_20260619_143632/replay_threshold/target_30/openvidhd_part1_057.mp4 |
| 5 | openvid_017 | openvidhd_part1_016 | 20 | online_adaptive | 170.567 | 535.343 | 3.139 | 20.344 | 0.6706 | 74 | 26 | 0.167877 | 100 |  | /hy-tmp/wan22_adaptive_seacache_overhead_train5_50step_45f_480p_20260619_143632/online_adaptive/target_20/openvidhd_part1_016.mp4 |
| 5 | openvid_017 | openvidhd_part1_016 | 20 | replay_threshold | 170.402 | 535.343 | 3.142 | 20.344 | 0.6706 | 74 | 26 |  | 0 | 0 | /hy-tmp/wan22_adaptive_seacache_overhead_train5_50step_45f_480p_20260619_143632/replay_threshold/target_20/openvidhd_part1_016.mp4 |
| 5 | openvid_017 | openvidhd_part1_016 | 25 | online_adaptive | 190.365 | 535.343 | 2.812 | 28.819 | 0.5804 | 70 | 30 | 0.183837 | 100 |  | /hy-tmp/wan22_adaptive_seacache_overhead_train5_50step_45f_480p_20260619_143632/online_adaptive/target_25/openvidhd_part1_016.mp4 |
| 5 | openvid_017 | openvidhd_part1_016 | 25 | replay_threshold | 190.163 | 535.343 | 2.815 | 28.819 | 0.5804 | 70 | 30 |  | 0 | 0 | /hy-tmp/wan22_adaptive_seacache_overhead_train5_50step_45f_480p_20260619_143632/replay_threshold/target_25/openvidhd_part1_016.mp4 |
| 5 | openvid_017 | openvidhd_part1_016 | 30 | online_adaptive | 239.543 | 535.343 | 2.235 | 33.833 | 0.3855 | 60 | 40 | 0.183311 | 100 |  | /hy-tmp/wan22_adaptive_seacache_overhead_train5_50step_45f_480p_20260619_143632/online_adaptive/target_30/openvidhd_part1_016.mp4 |
| 5 | openvid_017 | openvidhd_part1_016 | 30 | replay_threshold | 239.420 | 535.343 | 2.236 | 33.833 | 0.3855 | 60 | 40 |  | 0 | 0 | /hy-tmp/wan22_adaptive_seacache_overhead_train5_50step_45f_480p_20260619_143632/replay_threshold/target_30/openvidhd_part1_016.mp4 |

## 3. Prompt-Averaged Results

### 3.1 Train15/Test5: Average Across Prompts By Target

| target | n | psnr_mean | psnr_min | psnr_max | speedup_mean | speedup_min | speedup_max | elapsed_mean_s | thr_mean | reuse_mean | recompute_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20.0 | 20 | 21.108 | 15.416 | 33.857 | 3.171 | 2.975 | 3.381 | 171.498 | 0.5989 | 73.8 | 26.2 |
| 25.0 | 20 | 23.407 | 17.671 | 31.613 | 2.461 | 2.027 | 2.874 | 223.401 | 0.4156 | 63.2 | 36.8 |
| 30.0 | 20 | 27.221 | 19.993 | 34.824 | 1.904 | 1.544 | 2.491 | 291.516 | 0.2660 | 49.4 | 50.6 |

### 3.2 Train15/Test5: Average Across Prompts By Split And Target

| split | target | n | psnr_mean | psnr_min | psnr_max | speedup_mean | speedup_min | speedup_max | elapsed_mean_s | thr_mean | reuse_mean | recompute_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | 20.0 | 5 | 22.988 | 19.355 | 33.857 | 3.196 | 3.177 | 3.209 | 170.070 | 0.6117 | 74.0 | 26.0 |
| test | 25.0 | 5 | 24.219 | 21.175 | 31.125 | 2.595 | 2.275 | 2.874 | 210.769 | 0.4503 | 65.6 | 34.4 |
| test | 30.0 | 5 | 26.454 | 23.797 | 31.522 | 1.978 | 1.661 | 2.491 | 279.968 | 0.2770 | 51.6 | 48.4 |
| train | 20.0 | 15 | 20.481 | 15.416 | 29.135 | 3.163 | 2.975 | 3.381 | 171.974 | 0.5946 | 73.7 | 26.3 |
| train | 25.0 | 15 | 23.137 | 17.671 | 31.613 | 2.416 | 2.027 | 2.857 | 227.611 | 0.4040 | 62.4 | 37.6 |
| train | 30.0 | 15 | 27.477 | 19.993 | 34.824 | 1.879 | 1.544 | 2.368 | 295.366 | 0.2624 | 48.7 | 51.3 |

### 3.3 Train15/Test5: Per-Prompt Average Across Targets

| sample_id | source_id | split | n_targets | psnr_mean | speedup_mean | elapsed_mean_s | thr_mean | reuse_mean | recompute_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| openvid_086 | openvidhd_part1_085 | train | 3 | 24.813 | 2.801 | 197.344 | 0.5293 | 68.7 | 31.3 |
| openvid_087 | openvidhd_part1_086 | train | 3 | 22.311 | 2.516 | 226.842 | 0.3993 | 62.7 | 37.3 |
| openvid_060 | openvidhd_part1_059 | train | 3 | 25.613 | 2.362 | 246.784 | 0.3915 | 58.7 | 41.3 |
| openvid_058 | openvidhd_part1_057 | train | 3 | 31.546 | 2.811 | 197.282 | 0.5468 | 68.7 | 31.3 |
| openvid_017 | openvidhd_part1_016 | train | 3 | 27.665 | 2.734 | 199.823 | 0.5455 | 68.0 | 32.0 |
| openvid_037 | openvidhd_part1_036 | train | 3 | 28.560 | 2.707 | 206.513 | 0.4594 | 66.7 | 33.3 |
| openvid_094 | openvidhd_part1_093 | train | 3 | 22.968 | 2.221 | 262.368 | 0.3472 | 55.3 | 44.7 |
| openvid_064 | openvidhd_part1_063 | train | 3 | 21.351 | 2.495 | 229.497 | 0.4145 | 62.0 | 38.0 |
| openvid_096 | openvidhd_part1_095 | train | 3 | 22.399 | 2.440 | 236.432 | 0.4056 | 60.7 | 39.3 |
| openvid_059 | openvidhd_part1_058 | train | 3 | 19.802 | 2.364 | 248.561 | 0.3687 | 58.0 | 42.0 |
| openvid_028 | openvidhd_part1_027 | train | 3 | 24.752 | 2.263 | 255.687 | 0.3471 | 56.7 | 43.3 |
| openvid_013 | openvidhd_part1_012 | train | 3 | 19.772 | 2.198 | 262.630 | 0.3347 | 55.3 | 44.7 |
| openvid_021 | openvidhd_part1_020 | train | 3 | 24.896 | 2.688 | 209.732 | 0.4727 | 66.0 | 34.0 |
| openvid_032 | openvidhd_part1_031 | train | 3 | 19.051 | 2.414 | 239.285 | 0.3927 | 60.0 | 40.0 |
| openvid_038 | openvidhd_part1_037 | train | 3 | 19.972 | 2.276 | 255.975 | 0.3497 | 56.7 | 43.3 |
| openvid_029 | openvidhd_part1_028 | test | 3 | 32.168 | 2.856 | 192.769 | 0.5444 | 69.3 | 30.7 |
| openvid_031 | openvidhd_part1_030 | test | 3 | 22.584 | 2.660 | 212.098 | 0.4720 | 65.3 | 34.7 |
| openvid_027 | openvidhd_part1_026 | test | 3 | 21.754 | 2.424 | 234.714 | 0.3853 | 60.7 | 39.3 |
| openvid_093 | openvidhd_part1_092 | test | 3 | 22.351 | 2.455 | 238.650 | 0.4081 | 60.0 | 40.0 |
| openvid_056 | openvidhd_part1_055 | test | 3 | 23.911 | 2.551 | 223.115 | 0.4218 | 63.3 | 36.7 |

### 3.4 Overhead: Online/Replay Aggregate By Method And Target

| method | target | n | psnr_mean | speedup_mean | elapsed_mean_s | thr_mean | predictor_total_mean_s | decision_mismatch_sum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| online_adaptive | 20.0 | 5 | 22.904 | 3.207 | 169.166 | 0.6328 | 0.196965 |  |
| online_adaptive | 25.0 | 5 | 25.706 | 2.623 | 208.374 | 0.4880 | 0.188178 |  |
| online_adaptive | 30.0 | 5 | 30.559 | 2.103 | 263.189 | 0.3268 | 0.190274 |  |
| replay_threshold | 20.0 | 5 | 22.904 | 3.215 | 168.767 | 0.6328 |  | 0 |
| replay_threshold | 25.0 | 5 | 25.706 | 2.624 | 208.250 | 0.4880 |  | 0 |
| replay_threshold | 30.0 | 5 | 30.559 | 2.103 | 263.130 | 0.3268 |  | 0 |

### 3.5 Overhead: Paired Online Minus Replay Timing

| target | n_pairs | delta_mean_s | delta_min_s | delta_max_s | delta_pct_mean | predictor_total_mean_s | predictor_pct_compute_mean | decision_mismatch_sum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20 | 5 | 0.399600 | 0.143000 | 0.705000 | 0.238310 | 0.196965 | 0.116393 | 0 |
| 25 | 5 | 0.123800 | -0.035000 | 0.336000 | 0.057624 | 0.188178 | 0.090824 | 0 |
| 30 | 5 | 0.058800 | -0.167000 | 0.420000 | 0.017194 | 0.190274 | 0.074247 | 0 |
| all | 15 | 0.194067 | -0.167000 | 0.705000 | 0.104376 | 0.191806 | 0.093821 | 0 |

### 3.6 Overhead: Per-Prompt Paired Results

| order | sample_id | source_id | target | online_elapsed_s | replay_elapsed_s | delta_s | delta_pct | predictor_total_s | predictor_pct_compute | psnr_db | online_speedup | replay_speedup | decision_mismatch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | openvid_086 | openvidhd_part1_085 | 20 | 171.573 | 170.868 | 0.705 | 0.413 | 0.236853 | 0.1380 | 25.497 | 3.174 | 3.187 | 0 |
| 1 | openvid_086 | openvidhd_part1_085 | 25 | 190.627 | 190.636 | -0.009 | -0.005 | 0.175867 | 0.0923 | 25.432 | 2.857 | 2.857 | 0 |
| 1 | openvid_086 | openvidhd_part1_085 | 30 | 230.027 | 230.132 | -0.105 | -0.046 | 0.207156 | 0.0901 | 23.510 | 2.368 | 2.366 | 0 |
| 2 | openvid_087 | openvidhd_part1_086 | 20 | 171.059 | 170.916 | 0.143 | 0.084 | 0.178457 | 0.1043 | 18.500 | 3.186 | 3.189 | 0 |
| 2 | openvid_087 | openvidhd_part1_086 | 25 | 220.155 | 220.190 | -0.035 | -0.016 | 0.204076 | 0.0927 | 21.696 | 2.476 | 2.475 | 0 |
| 2 | openvid_087 | openvidhd_part1_086 | 30 | 288.661 | 288.828 | -0.167 | -0.058 | 0.197612 | 0.0685 | 26.736 | 1.888 | 1.887 | 0 |
| 3 | openvid_060 | openvidhd_part1_059 | 20 | 171.092 | 170.710 | 0.382 | 0.224 | 0.215884 | 0.1262 | 21.045 | 3.176 | 3.183 | 0 |
| 3 | openvid_060 | openvidhd_part1_059 | 25 | 240.003 | 239.667 | 0.336 | 0.140 | 0.187398 | 0.0781 | 20.972 | 2.264 | 2.267 | 0 |
| 3 | openvid_060 | openvidhd_part1_059 | 30 | 328.533 | 328.113 | 0.420 | 0.128 | 0.169179 | 0.0515 | 34.824 | 1.654 | 1.656 | 0 |
| 4 | openvid_058 | openvidhd_part1_057 | 20 | 161.540 | 160.937 | 0.603 | 0.375 | 0.185754 | 0.1150 | 29.135 | 3.361 | 3.374 | 0 |
| 4 | openvid_058 | openvidhd_part1_057 | 25 | 200.718 | 200.593 | 0.125 | 0.062 | 0.189711 | 0.0945 | 31.613 | 2.705 | 2.707 | 0 |
| 4 | openvid_058 | openvidhd_part1_057 | 30 | 229.182 | 229.159 | 0.023 | 0.010 | 0.194114 | 0.0847 | 33.890 | 2.369 | 2.369 | 0 |
| 5 | openvid_017 | openvidhd_part1_016 | 20 | 170.567 | 170.402 | 0.165 | 0.097 | 0.167877 | 0.0984 | 20.344 | 3.139 | 3.142 | 0 |
| 5 | openvid_017 | openvidhd_part1_016 | 25 | 190.365 | 190.163 | 0.202 | 0.106 | 0.183837 | 0.0966 | 28.819 | 2.812 | 2.815 | 0 |
| 5 | openvid_017 | openvidhd_part1_016 | 30 | 239.543 | 239.420 | 0.123 | 0.051 | 0.183311 | 0.0765 | 33.833 | 2.235 | 2.236 | 0 |

## 4. Simple Conclusions

- The adaptive threshold predictor controls SeaCache in the expected direction: higher target PSNR lowers the mean predicted threshold, reduces reuse, improves PSNR, and reduces speedup.
- On the 20-prompt train15/test5 run, target 20/25/30 achieved mean PSNR of 21.108/23.407/27.221 dB with mean speedup of 3.171x/2.461x/1.904x. The ordering is correct, but absolute PSNR is below the requested target on average, especially for target 25 and target 30.
- The held-out 5 test prompts are not worse than train prompts in this sample. Test means are 22.988/24.219/26.454 dB at targets 20/25/30, while train means are 20.481/23.137/27.477 dB. The small test sample makes this indicative rather than conclusive.
- There are substantial prompt-level outliers. Some prompts fall well below the nominal target even when target_psnr is 25 or 30, so the predictor is not yet calibrated as a reliable per-prompt quality controller.
- The overhead experiment shows that predictor overhead is negligible. Online adaptive minus replay averages 0.194 seconds per candidate, about 0.104% of replay compute time. Direct measured predictor time averages 0.192 seconds per candidate, about 0.094% of online compute time.
- Online and replay traces have zero decision mismatches across 15 prompt-target pairs, so replay is a valid control for measuring predictor overhead in this setup.

## Source Files

| Artifact | Path |
| --- | --- |
| train15/test5 summary CSV | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/results/summary.csv |
| train15/test5 summary JSON | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/results/summary.json |
| overhead summary CSV | /hy-tmp/wan22_adaptive_seacache_overhead_train5_50step_45f_480p_20260619_143632/results/summary.csv |
| overhead summary JSON | /hy-tmp/wan22_adaptive_seacache_overhead_train5_50step_45f_480p_20260619_143632/results/summary.json |
| train15/test5 runner resume log | /hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/logs |
| overhead runner resume log | /hy-tmp/wan22_adaptive_seacache_overhead_train5_50step_45f_480p_20260619_143632/logs/runner_resume_20260623.log |
