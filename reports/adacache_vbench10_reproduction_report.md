# AdaCache Wan2.2-A14B VBench10 Reproduction Report

Generated: 2026-06-23 Asia/Shanghai

## 1. Environment, Parameters, And Configuration

### Experiment Archive

- Experiment root: `/hy-tmp/wan22_adacache_vbench10_subprocess_50step_45f_480p_20260619_155622`
- Combined result CSV: `/hy-tmp/wan22_adacache_vbench10_subprocess_50step_45f_480p_20260619_155622/results/summary_all.csv`
- Combined aggregate JSON: `/hy-tmp/wan22_adacache_vbench10_subprocess_50step_45f_480p_20260619_155622/results/aggregate_all.json`
- Shard summaries:
  - `/hy-tmp/wan22_adacache_vbench10_subprocess_50step_45f_480p_20260619_155622/results/summary_shard_0.csv`
  - `/hy-tmp/wan22_adacache_vbench10_subprocess_50step_45f_480p_20260619_155622/results/summary_shard_1.csv`
- Report copy in experiment archive: `/hy-tmp/wan22_adacache_vbench10_subprocess_50step_45f_480p_20260619_155622/results/adacache_vbench10_reproduction_report.md`

### Machine

- Host OS: Ubuntu Linux, kernel `5.15.0-181-generic`, x86_64
- CPU: AMD EPYC 7V13 64-Core Processor, 1 socket, 64 cores / 128 threads
- System memory: 251 GiB
- Data filesystem: `/hy-tmp`, 600 GiB total, 325 GiB available at report generation
- GPUs during run: 2 x NVIDIA RTX PRO 6000 Blackwell Workstation Edition
- Per-GPU memory during run: 97,887 MiB reported by `nvidia-smi`
- Driver during run: 580.167.08
- CUDA reported by driver during run: 13.0

Note: on 2026-06-23 this container/session reports `nvidia-smi: No devices were found`; the GPU details above are taken from the archived run-time `gpu_*.txt` files in the experiment directory.

### Software Environment

- Python environment: `/hy-tmp/env/Wan2.2-fa2torch28`
- Python: 3.10.12
- PyTorch: 2.8.0+cu128
- PyTorch CUDA runtime: 12.8
- FlashAttention: 2.8.3.post1
- FFmpeg / FFprobe: 8.1.1 from the same environment
- Repository: `/hy-tmp/work/Wan2.2`
- Git revision at report generation: `b5ab7a3` with local experiment changes present

### Model And Data

- Task/model: `t2v-A14B`
- Checkpoint: `/hy-tmp/models/Wan2.2-T2V-A14B`
- Prompt set: `/hy-tmp/work/Wan2.2/test_sets/Vbench10/prompts.jsonl`
- Number of prompts: 10
- Sharding: prompt 1-5 on GPU 0, prompt 6-10 on GPU 1

### Generation Parameters

- Resolution: `832*480`
- Frames: 45
- FPS: 16
- Video duration: 2.8125 s
- Sampling steps: 50
- Sampler: `dpm++`
- Seed: 42
- Shift: 12.0
- Guide scale: `(3.0, 4.0)`
- `offload_model`: True
- `convert_model_dtype`: True
- Timing metric: `inference_compute_elapsed_seconds` parsed from generation logs. Model loading, process startup, video saving, and PSNR computation are not included in speedup timing.
- PSNR metric: FFmpeg `psnr` filter, YUV weighted average; perfect/infinite frames excluded when present. In this run, no perfect frames were excluded.

### AdaCache Configuration

Shared settings:

- Cache residual type: `t-attn`
- Cache location for schedule metric: `13`
- MoReg: disabled
- Cache dtype: `input`
- Separate high/low codebook arguments are supported and were set explicitly.
- Runtime mode: method-isolated subprocesses with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

Reason for subprocess mode: full 45-frame AdaCache generation OOMed when running `baseline -> slow -> fast` in a single Python/CUDA process due to CUDA allocator retained segments and AdaCache residual-cache peak memory. Running each method in a fresh process preserves AdaCache semantics while isolating allocator state; a 50-step/45-frame smoke test confirmed this avoids OOM.

Codebooks:

| config | high codebook | low codebook |
| --- | --- | --- |
| slow | `0.066:4,0.129:3,0.191:2,0.254:1,0.317:1,1.0:1` | `0.066:4,0.129:3,0.191:2,0.254:1,0.317:1,1.0:1` |
| fast | `0.066:8,0.129:6,0.191:5,0.254:4,0.317:3,1.0:2` | `0.066:8,0.129:6,0.191:5,0.254:4,0.317:3,1.0:2` |

### Completion And Validation

| item | count |
| --- | ---: |
| baseline videos | 10 / 10 |
| slow videos | 10 / 10 |
| fast videos | 10 / 10 |
| PSNR json files | 20 / 20 |
| completed result rows | 20 / 20 |
| failed files | 0 |
| ffprobe frame/size mismatches | 0 |

## 2. Complete Experimental Result Table

The table below is a compact complete result table with one row per prompt and AdaCache configuration. Full raw columns, including video paths, ffprobe JSON, reuse/recompute paths, diff paths, and rate paths, are available in `/hy-tmp/wan22_adacache_vbench10_subprocess_50step_45f_480p_20260619_155622/results/summary_all.csv`.

| sample_id | method | prompt_short | baseline_s | adacache_s | speedup_x | mean_psnr | min_psnr | max_psnr | frames | reuse | recompute |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vbench10_001 | fast | A woman is playing football. | 329.470 | 125.657 | 2.622 | 17.644 | 15.380 | 19.440 | 45 | 69 | 31 |
| vbench10_001 | slow | A woman is playing football. | 329.470 | 221.812 | 1.485 | 26.257 | 21.550 | 30.920 | 45 | 37 | 63 |
| vbench10_002 | fast | A horse is running along the beach, then it suddenly stops and starts grazing. | 329.764 | 119.676 | 2.755 | 19.746 | 15.650 | 22.000 | 45 | 71 | 29 |
| vbench10_002 | slow | A horse is running along the beach, then it suddenly stops and starts grazing. | 329.764 | 203.747 | 1.618 | 26.230 | 16.890 | 29.600 | 45 | 43 | 57 |
| vbench10_003 | fast | The race began, and Team A quickly took the lead, securing the front position. ... | 329.957 | 122.659 | 2.690 | 15.511 | 14.600 | 16.730 | 45 | 70 | 30 |
| vbench10_003 | slow | The race began, and Team A quickly took the lead, securing the front position. ... | 329.957 | 221.846 | 1.487 | 21.194 | 18.770 | 23.360 | 45 | 37 | 63 |
| vbench10_004 | fast | Snow White was driven into the forest by the evil queen who was jealous of her ... | 329.750 | 125.627 | 2.625 | 18.131 | 16.620 | 19.650 | 45 | 69 | 31 |
| vbench10_004 | slow | Snow White was driven into the forest by the evil queen who was jealous of her ... | 329.750 | 245.761 | 1.342 | 20.344 | 18.600 | 22.330 | 45 | 29 | 71 |
| vbench10_005 | fast | The camera orbits around in a clockwise direction. Forbidden City. | 329.717 | 119.603 | 2.757 | 15.563 | 14.370 | 16.240 | 45 | 71 | 29 |
| vbench10_005 | slow | The camera orbits around in a clockwise direction. Forbidden City. | 329.717 | 194.676 | 1.694 | 16.731 | 15.750 | 17.680 | 45 | 46 | 54 |
| vbench10_006 | fast | Equal amounts of yellow and blue paint are rapidly combined, with the mixture b... | 336.123 | 124.710 | 2.695 | 18.212 | 16.740 | 20.360 | 45 | 70 | 30 |
| vbench10_006 | slow | Equal amounts of yellow and blue paint are rapidly combined, with the mixture b... | 336.123 | 207.595 | 1.619 | 21.463 | 19.360 | 23.270 | 45 | 43 | 57 |
| vbench10_007 | fast | A timelapse captures the reaction as concentrated sulfuric acid is poured onto ... | 336.745 | 124.684 | 2.701 | 21.108 | 19.620 | 23.260 | 45 | 70 | 30 |
| vbench10_007 | slow | A timelapse captures the reaction as concentrated sulfuric acid is poured onto ... | 336.745 | 225.993 | 1.490 | 27.835 | 25.510 | 32.550 | 45 | 37 | 63 |
| vbench10_008 | fast | A cat is on the right of a rock, then the cat runs to the left of the rock. | 336.671 | 121.709 | 2.766 | 20.923 | 19.260 | 22.220 | 45 | 71 | 29 |
| vbench10_008 | slow | A cat is on the right of a rock, then the cat runs to the left of the rock. | 336.671 | 207.470 | 1.623 | 21.946 | 20.710 | 23.030 | 45 | 43 | 57 |
| vbench10_009 | fast | people are playing ping-pong. | 336.911 | 127.788 | 2.636 | 17.061 | 15.260 | 19.530 | 45 | 69 | 31 |
| vbench10_009 | slow | people are playing ping-pong. | 336.911 | 263.059 | 1.281 | 27.847 | 24.640 | 30.240 | 45 | 25 | 75 |
| vbench10_010 | fast | The camera orbits around. Serengeti, the camera circles around. | 336.408 | 121.434 | 2.770 | 22.451 | 21.710 | 23.670 | 45 | 71 | 29 |
| vbench10_010 | slow | The camera orbits around. Serengeti, the camera circles around. | 336.408 | 185.840 | 1.810 | 25.761 | 24.610 | 26.370 | 45 | 50 | 50 |

## 3. Aggregate Table Over 10 Prompts

| method | num_prompts | mean_baseline_s | mean_adacache_s | mean_speedup_x | speedup_range_x | mean_psnr | mean_psnr_range | mean_min_psnr | mean_reuse | mean_recompute |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| slow | 10 | 333.152 | 217.780 | 1.545 | 1.281-1.810 | 23.561 | 16.731-27.847 | 20.639 | 39.000 | 61.000 |
| fast | 10 | 333.152 | 123.355 | 2.702 | 2.622-2.770 | 18.635 | 15.511-22.451 | 16.921 | 70.100 | 29.900 |

## 4. Conclusion And Summary

- The AdaCache reproduction on Wan2.2-A14B completed successfully on VBench10: all 10 baseline videos, 10 slow videos, and 10 fast videos were generated and evaluated.
- The slow configuration achieved an average speedup of `1.545x` with average PSNR `23.561`.
- The fast configuration achieved an average speedup of `2.702x` with average PSNR `18.635`.
- The expected speed-quality tradeoff is visible: fast reuses more computation and is substantially faster, but has lower PSNR; slow is more conservative and preserves higher PSNR.
- All evaluated videos passed ffprobe validation at 832x480, 45 frames, 16 fps, and no PSNR calculation failures occurred.
- No failed samples were recorded in the final subprocess-mode run.

Caveats:

- The 50-step Wan2.2 codebooks are interpolated/derived from the public AdaCache 30-step and 100-step presets because no official Wan2.2 50-step codebook was available.
- Method-isolated subprocess execution was used to avoid CUDA allocator retention/OOM at 45 frames. This changes process scheduling overhead but not the measured compute-only generation timing used for speedup.
- This report focuses on PSNR relative to the same-prompt/seed no-cache baseline; human perceptual quality and VBench semantic metrics were not computed here.
