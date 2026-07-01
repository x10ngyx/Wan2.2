# SeaCache Wan2.1 vs Wan2.2 Ali-10 UniPC Report

Date: 2026-06-30

This report records two Ali-10 SeaCache experiments:

- Wan2.1 SeaCache Ali-10 threshold compare from `/hy-tmp/work/Wan2.1-seacache-ali10`
- Wan2.2 Ali-10 UniPC timestep-only SeaCache rerun from `/hy-tmp/work/Wan2.2`

The purpose is to document the evidence for the working conclusion that the observed SeaCache performance gap is primarily caused by Wan2.1 vs Wan2.2 model / architecture / inference-path differences, rather than by the previously mixed sampler setting.

## Executive Summary

After rerunning Wan2.2 with the same high-level Ali-10 setting used by Wan2.1, the performance gap remains:

| Experiment | Model | Sampler | Prompt set | Threshold | Speedup | Mean PSNR |
| --- | --- | --- | --- | ---: | ---: | ---: |
| Wan2.1 SeaCache threshold compare | Wan2.1 T2V-14B | UniPC | Ali-10 | `0.20` | `1.8588x` | `30.1260 dB` |
| Wan2.2 timestep-only SeaCache rerun | Wan2.2 T2V-A14B | UniPC | Ali-10 | `0.20` | `1.5760x` | `27.5130 dB` |
| Wan2.2 - Wan2.1 | - | - | - | - | `-0.2828x` | `-2.6130 dB` |

The important point is that Wan2.2 is worse in PSNR even though it is also less accelerated. This means the lower PSNR is not explained by Wan2.2 skipping more computation. The sampler confound has also been controlled: both experiments use UniPC on Ali-10 with seed `42`, `832*480`, `45` frames, and `50` steps.

This supports the interpretation that the quality gap is mainly due to Wan2.2 model / architecture / inference-path differences, including the Wan2.2 high/low-stage execution path. It is not a strict mathematical proof that only model weights are responsible, because Wan2.2 also differs in high/low stage scheduling, guidance-scale form, and sigma/shift behavior. The Wan2.1 reference is `seacache_thresh=0.0`, not a separately implemented no-cache baseline, so this should be noted when citing the result.

## Method Overview

SeaCache is a timestep cache method inside the model forward path. It is not a guided-output-level cache.

At a high level:

1. For each denoising step, the method extracts an early transformer feature, specifically around the first block's timestep-modulated normalized input.
2. It applies the SeaCache frequency-domain filtering / smoothing logic to the feature.
3. It compares the accumulated relative-L1 feature difference against a threshold.
4. If the accumulated difference is below threshold, it reuses cached transformer-block residual output for that branch/stage.
5. If the accumulated difference exceeds threshold, it recomputes the transformer blocks and refreshes the cache.
6. The final head / unpatchify path still runs; the cache is for the expensive transformer-block residual path.

Both implementations cache inside cond/uncond branches, not after CFG guidance:

- Wan2.1 SeaCache uses the original SeaCache-style monkey patch and separates cond/uncond by call order state.
- Wan2.2 SeaCache uses an explicit `SeaCacheTimestepCache` object and keys cache state by `(model_stage, branch)`, e.g. `('high', 'cond')`, `('low', 'uncond')`.

The Wan2.2 high/low split is a remaining model-path difference: the same threshold is applied, but cache state is bucketed by model stage and branch.

## Shared Evaluation Setup

| Item | Value |
| --- | --- |
| Prompt set | Ali-10 |
| Prompt file | `/hy-tmp/work/Wan2.2/test_sets/ali_10/prompts.jsonl` |
| Sample IDs | `ali_001` ... `ali_010` |
| Seed | `42` |
| Resolution | `832*480` |
| Frames | `45` |
| Sampling steps | `50` |
| Sampler | `unipc` |
| Candidate SeaCache threshold | `0.20` |
| PSNR script | `/hy-tmp/work/compute_psnr.py` |
| PSNR metric | ffmpeg PSNR filter, weighted YUV average, excluding perfect frames above threshold |
| Timing target | inference / generation compute time; model loading and video saving are excluded by the experiment-specific timing source |

## Experiment 1: Wan2.1 SeaCache Ali-10 Threshold Compare

### Source and Outputs

| Item | Path / Value |
| --- | --- |
| Code root | `/hy-tmp/work/Wan2.1-seacache-ali10` |
| Runner | `/hy-tmp/work/Wan2.1-seacache-ali10/scripts/run_ali10_seacache_threshold_compare.py` |
| Python | `/hy-tmp/env/Wan2.1/bin/python` |
| Model checkpoint | `/hy-tmp/models/Wan2.1-T2V-14B` |
| Output directory | `/hy-tmp/wan21_seacache_threshold_compare_ali10_seed42_832x480_45f_50steps` |
| Config files | `experiment_config_gpu0.json`, `experiment_config_gpu1.json` |
| Summary CSV | `/hy-tmp/wan21_seacache_threshold_compare_ali10_seed42_832x480_45f_50steps/results/summary_all.csv` |
| Summary JSON | `/hy-tmp/wan21_seacache_threshold_compare_ali10_seed42_832x480_45f_50steps/results/summary_all.json` |
| Aggregate JSON | `/hy-tmp/wan21_seacache_threshold_compare_ali10_seed42_832x480_45f_50steps/results/aggregate.json` |
| PSNR directory | `/hy-tmp/wan21_seacache_threshold_compare_ali10_seed42_832x480_45f_50steps/psnr` |

### Complete Configuration

| Config item | Value |
| --- | --- |
| `runner` | `/hy-tmp/work/Wan2.1-seacache-ali10/scripts/run_ali10_seacache_threshold_compare.py` |
| `root` | `/hy-tmp/work/Wan2.1-seacache-ali10` |
| `python_bin` | `/hy-tmp/env/Wan2.1/bin/python` |
| `compute_psnr` | `/hy-tmp/work/compute_psnr.py` |
| `prompt_path` | `/hy-tmp/work/Wan2.2/test_sets/ali_10/prompts.jsonl` |
| `output_dir` | `/hy-tmp/wan21_seacache_threshold_compare_ali10_seed42_832x480_45f_50steps` |
| `ckpt_dir` | `/hy-tmp/models/Wan2.1-T2V-14B` |
| `task` | `t2v-14B` |
| `size` | `832*480` |
| `frame_num` | `45` |
| `sample_steps` | `50` |
| `base_seed` | `42` |
| `reference_thresh` | `0.0` |
| `candidate_thresh` | `0.2` |
| `reference_definition` | `SeaCache Wan2.1 with seacache_thresh=0` |
| `candidate_definition` | `SeaCache Wan2.1 official default seacache_thresh=0.2` |
| `timing_definition` | `INFERENCE_TIME_SECONDS` measured inside `seacache_generate.py` around `wan_t2v.generate()`, excluding model loading and video saving |
| `sampler_left_default` | `unipc` |
| Official options left at defaults | `offload_model`, `sample_solver`, `sample_shift`, `sample_guide_scale`, `use_ret_steps`, `t5_cpu`, `t5_fsdp`, `dit_fsdp`, `ulysses_size`, `ring_size`, `prompt_extend` |
| GPU / shard files | `experiment_config_gpu0.json`, `experiment_config_gpu1.json`; their `start` / `limit` fields reflect the shard/resume invocations, while `summary_all.csv` is the merged 10-prompt result |

### Aggregate Result

| Metric | Value |
| --- | ---: |
| Completed pairs | `10` |
| Mean speedup | `1.8588x` |
| Min speedup | `1.8467x` |
| Max speedup | `1.8692x` |
| Mean PSNR | `30.1260 dB` |
| Min prompt mean PSNR | `23.4300 dB` |
| Max prompt mean PSNR | `40.7540 dB` |
| Mean reference inference time | `541.0653s` |
| Mean candidate inference time | `291.0806s` |
| Total reference inference time | `5410.6529s` |
| Total candidate inference time | `2910.8063s` |
| Overall speedup from totals | `1.8588x` |

### Per-Prompt Result

| Sample | Reference time | Candidate time | Speedup | Mean PSNR | Min PSNR | Max PSNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ali_001` | `537.240s` | `287.419s` | `1.8692x` | `27.8998` | `25.02` | `29.10` |
| `ali_002` | `540.441s` | `290.773s` | `1.8586x` | `40.7540` | `38.14` | `41.95` |
| `ali_003` | `541.403s` | `293.166s` | `1.8467x` | `32.6547` | `30.27` | `35.82` |
| `ali_004` | `539.316s` | `290.860s` | `1.8542x` | `26.7629` | `25.67` | `27.91` |
| `ali_005` | `538.806s` | `290.011s` | `1.8579x` | `40.3633` | `39.84` | `41.20` |
| `ali_006` | `540.864s` | `290.709s` | `1.8605x` | `23.4300` | `21.75` | `25.33` |
| `ali_007` | `542.304s` | `292.767s` | `1.8523x` | `32.7887` | `30.47` | `35.53` |
| `ali_008` | `547.581s` | `294.673s` | `1.8583x` | `25.5924` | `22.65` | `28.94` |
| `ali_009` | `542.596s` | `290.518s` | `1.8677x` | `27.1311` | `25.66` | `27.90` |
| `ali_010` | `540.101s` | `289.910s` | `1.8630x` | `23.8831` | `21.54` | `27.23` |

## Experiment 2: Wan2.2 Ali-10 UniPC Timestep-Only SeaCache

### Source and Outputs

| Item | Path / Value |
| --- | --- |
| Code root | `/hy-tmp/work/Wan2.2` |
| Runner | `/hy-tmp/work/Wan2.2/experiments/seacache_ali10_unipc_50step_45f_480p/run_batch.py` |
| Launcher | `/hy-tmp/work/Wan2.2/experiments/seacache_ali10_unipc_50step_45f_480p/run_tmux.sh` |
| Python | `/hy-tmp/env/Wan2.2/bin/python` |
| Model checkpoint | `/hy-tmp/models/Wan2.2-T2V-A14B` |
| Output directory | `/hy-tmp/wan22_seacache_ali10_unipc_50step_45f_480p_20260629_224430` |
| Config file | `/hy-tmp/wan22_seacache_ali10_unipc_50step_45f_480p_20260629_224430/experiment_config.json` |
| Summary CSV | `/hy-tmp/wan22_seacache_ali10_unipc_50step_45f_480p_20260629_224430/results/summary.csv` |
| Summary JSON | `/hy-tmp/wan22_seacache_ali10_unipc_50step_45f_480p_20260629_224430/results/summary.json` |
| Aggregate JSON | `/hy-tmp/wan22_seacache_ali10_unipc_50step_45f_480p_20260629_224430/results/aggregate_by_threshold.json` |
| PSNR directory | `/hy-tmp/wan22_seacache_ali10_unipc_50step_45f_480p_20260629_224430/psnr/th_0p20` |

### Complete Configuration

| Config item | Value |
| --- | --- |
| `root_dir` | `/hy-tmp/work/Wan2.2` |
| `python_bin` | `/hy-tmp/env/Wan2.2/bin/python` |
| `ckpt_dir` | `/hy-tmp/models/Wan2.2-T2V-A14B` |
| `prompt_path` | `/hy-tmp/work/Wan2.2/test_sets/ali_10/prompts.jsonl` |
| `exp_root` | `/hy-tmp/wan22_seacache_ali10_unipc_50step_45f_480p_20260629_224430` |
| `task` | `t2v-A14B` |
| `size` | `832*480` |
| `frame_num` | `45` |
| `sample_steps` | `50` |
| `sample_solver` | `unipc` |
| `sample_shift` | `12.0` |
| `sample_guide_scale` | `[3.0, 4.0]` |
| `base_seed` | `42` |
| `thresholds` | `0.20` |
| `prompt_start` | `0` |
| `prompt_limit` | `0`, meaning all selected prompts |
| `selected_prompt_count` | `10` |
| `expected_baseline_runs` | `10` |
| `expected_candidate_runs` | `10` |
| `offload_model` | `true` |
| `convert_model_dtype` | `true` |
| `resume_existing` | `true` |
| `cpu_validate` | `false` |
| `ffprobe_bin` | `/hy-tmp/env/Wan2.2/bin/ffprobe` |
| `psnr_script` | `/hy-tmp/work/compute_psnr.py` |
| `timestep_cache` | `seacache` |
| `block_cache` | `none` |
| `cfg_cache` | `none` |
| `seacache_num_steps` | `null` |
| `seacache_use_ret_steps` | `false` |
| `seacache_power_exp` | `3.0` |
| `seacache_power_const` | `1.0` |
| `seacache_eps` | `1e-16` |
| `seacache_norm_mode` | `mean` |
| `timing_source` | `inference_compute_elapsed_seconds` |

### Aggregate Result

| Metric | Value |
| --- | ---: |
| Completed pairs | `10` |
| Threshold | `0.20` |
| Total baseline elapsed | `5421.520s` |
| Total SeaCache elapsed | `3440.106s` |
| Overall speedup | `1.5760x` |
| Mean-of-mean PSNR | `27.5130 dB` |
| Min prompt mean PSNR | `22.4287 dB` |
| Global min PSNR | `21.04 dB` |
| Total SeaCache reuse count | `400` |
| Total SeaCache recompute count | `600` |

Each prompt had the same aggregate Wan2.2 SeaCache reuse/recompute count: `40/60`, split as high-stage cond/uncond `10/22` each and low-stage cond/uncond `10/8` each.

### Per-Prompt Result

| Sample | Baseline time | SeaCache time | Speedup | Mean PSNR | Min PSNR | Max PSNR | Reuse/Recompute |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ali_001` | `541.223s` | `344.302s` | `1.5719x` | `23.9542` | `22.51` | `26.44` | `40/60` |
| `ali_002` | `542.542s` | `344.006s` | `1.5771x` | `29.7913` | `25.95` | `30.93` | `40/60` |
| `ali_003` | `542.246s` | `343.966s` | `1.5765x` | `31.5704` | `30.73` | `32.48` | `40/60` |
| `ali_004` | `542.319s` | `343.834s` | `1.5773x` | `23.6969` | `21.04` | `24.48` | `40/60` |
| `ali_005` | `542.082s` | `343.816s` | `1.5767x` | `39.6138` | `38.51` | `40.98` | `40/60` |
| `ali_006` | `541.883s` | `344.160s` | `1.5745x` | `23.2389` | `21.74` | `24.10` | `40/60` |
| `ali_007` | `542.224s` | `343.750s` | `1.5774x` | `30.3627` | `28.50` | `31.21` | `40/60` |
| `ali_008` | `542.204s` | `344.101s` | `1.5757x` | `24.5182` | `21.53` | `26.18` | `40/60` |
| `ali_009` | `542.399s` | `344.321s` | `1.5753x` | `22.4287` | `21.35` | `23.25` | `40/60` |
| `ali_010` | `542.398s` | `343.850s` | `1.5774x` | `25.9553` | `24.10` | `27.84` | `40/60` |

## Cross-Experiment Comparison

### Aggregate Comparison

| Model / path | Reference definition | Candidate definition | Reference total | Candidate total | Speedup | Mean PSNR |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Wan2.1 | `seacache_thresh=0.0` | `seacache_thresh=0.2` | `5410.653s` | `2910.806s` | `1.8588x` | `30.1260 dB` |
| Wan2.2 | no-cache baseline | timestep-only SeaCache `threshold=0.20` | `5421.520s` | `3440.106s` | `1.5760x` | `27.5130 dB` |
| Wan2.2 - Wan2.1 | - | - | `+10.867s` | `+529.300s` | `-0.2828x` | `-2.6130 dB` |

### Per-Prompt Comparison

| Sample | Wan2.1 speedup | Wan2.2 speedup | Speedup delta | Wan2.1 PSNR | Wan2.2 PSNR | PSNR delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ali_001` | `1.8692x` | `1.5719x` | `-0.2972x` | `27.8998` | `23.9542` | `-3.9456` |
| `ali_002` | `1.8586x` | `1.5771x` | `-0.2815x` | `40.7540` | `29.7913` | `-10.9627` |
| `ali_003` | `1.8467x` | `1.5765x` | `-0.2703x` | `32.6547` | `31.5704` | `-1.0842` |
| `ali_004` | `1.8542x` | `1.5773x` | `-0.2769x` | `26.7629` | `23.6969` | `-3.0660` |
| `ali_005` | `1.8579x` | `1.5767x` | `-0.2812x` | `40.3633` | `39.6138` | `-0.7496` |
| `ali_006` | `1.8605x` | `1.5745x` | `-0.2860x` | `23.4300` | `23.2389` | `-0.1911` |
| `ali_007` | `1.8523x` | `1.5774x` | `-0.2750x` | `32.7887` | `30.3627` | `-2.4260` |
| `ali_008` | `1.8583x` | `1.5757x` | `-0.2826x` | `25.5924` | `24.5182` | `-1.0742` |
| `ali_009` | `1.8677x` | `1.5753x` | `-0.2924x` | `27.1311` | `22.4287` | `-4.7024` |
| `ali_010` | `1.8630x` | `1.5774x` | `-0.2856x` | `23.8831` | `25.9553` | `+2.0722` |

### Interpretation

The controlled Wan2.1 vs Wan2.2 SeaCache comparison uses:

- Ali-10 prompt set
- UniPC sampler
- seed `42`
- `832*480`
- `45` frames
- `50` steps
- SeaCache threshold `0.20`
- timestep-only SeaCache, with block cache and CFG cache disabled

After this control, Wan2.2 is still lower in mean PSNR by `2.6130 dB`.

Wan2.2 also achieves less speedup: `1.5760x` vs Wan2.1 `1.8588x`. Therefore the Wan2.2 PSNR drop is not explained by Wan2.2 using a more aggressive cache policy. In fact, Wan2.2 is less accelerated and still lower quality.

This pattern supports the conclusion that the performance difference is driven mainly by Wan2.1 vs Wan2.2 model / architecture / inference-path differences. The most relevant remaining implementation-side difference is that Wan2.2 has high/low model stages and stores SeaCache state by `(stage, branch)`, while Wan2.1 uses the original single-stage branch-call structure. That stage split is itself part of the Wan2.2 inference architecture/path.

### Remaining Caveats

This result supports the model/path-difference explanation, but it does not prove that only model weights are responsible. The following residual differences remain:

- Wan2.2 uses high/low stage scheduling and per-stage SeaCache state buckets.
- Wan2.2 uses `sample_shift=12.0` and `sample_guide_scale=[3.0, 4.0]`; Wan2.1 left these options at official defaults.
- Wan2.1 reference is `seacache_thresh=0.0`, not a separately implemented no-cache baseline. It should be very close to a no-reuse path, but it is still the SeaCache entry path.
- Wan2.1 summary does not include detailed reuse/recompute trace, while Wan2.2 records `400/600` total reuse/recompute.

Within these limits, the experiment supports Wan2.2 model / architecture / inference-path differences as the primary explanation for the observed SeaCache performance gap.

## Files to Reproduce or Audit

Wan2.1:

- Config: `/hy-tmp/wan21_seacache_threshold_compare_ali10_seed42_832x480_45f_50steps/experiment_config_gpu0.json`
- Config: `/hy-tmp/wan21_seacache_threshold_compare_ali10_seed42_832x480_45f_50steps/experiment_config_gpu1.json`
- Summary: `/hy-tmp/wan21_seacache_threshold_compare_ali10_seed42_832x480_45f_50steps/results/summary_all.csv`
- Aggregate: `/hy-tmp/wan21_seacache_threshold_compare_ali10_seed42_832x480_45f_50steps/results/aggregate.json`
- PSNR: `/hy-tmp/wan21_seacache_threshold_compare_ali10_seed42_832x480_45f_50steps/psnr/ali_XXX.json`
- Videos: `/hy-tmp/wan21_seacache_threshold_compare_ali10_seed42_832x480_45f_50steps/th_0p00/ali_XXX.mp4`
- Videos: `/hy-tmp/wan21_seacache_threshold_compare_ali10_seed42_832x480_45f_50steps/th_0p20/ali_XXX.mp4`

Wan2.2:

- Config: `/hy-tmp/wan22_seacache_ali10_unipc_50step_45f_480p_20260629_224430/experiment_config.json`
- Summary: `/hy-tmp/wan22_seacache_ali10_unipc_50step_45f_480p_20260629_224430/results/summary.csv`
- Aggregate: `/hy-tmp/wan22_seacache_ali10_unipc_50step_45f_480p_20260629_224430/results/aggregate_by_threshold.json`
- PSNR: `/hy-tmp/wan22_seacache_ali10_unipc_50step_45f_480p_20260629_224430/psnr/th_0p20/ali_XXX.json`
- Baseline videos: `/hy-tmp/wan22_seacache_ali10_unipc_50step_45f_480p_20260629_224430/baseline/ali_XXX.mp4`
- Candidate videos: `/hy-tmp/wan22_seacache_ali10_unipc_50step_45f_480p_20260629_224430/seacache/th_0p20/ali_XXX.mp4`
- Logs: `/hy-tmp/wan22_seacache_ali10_unipc_50step_45f_480p_20260629_224430/logs`
