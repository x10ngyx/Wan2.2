# Progress

## 2026-06-13 Initialization Snapshot

This file has been reset from the old long-form session history. Historical details were reviewed and condensed into this initialization state. Use this file as the current handoff source going forward.

## Current Goal

The project is a Wan2.2 T2V-14B inference-acceleration study with three stages:

1. Implement three threshold-based cache methods for inference acceleration:
   - timestep cache
   - block cache
   - CFG cache
2. Generate data mapping threshold combinations to video quality and speed:
   - threshold combination -> PSNR / speedup / reuse statistics / failure state
3. Build adaptive inference acceleration:
   - train a small predictor that takes target quality and speed requirements and predicts a threshold combination for inference.

## Environment And Resources

- Workspace: `/hy-tmp/work/Wan2.2`
- Conda environment: `/hy-tmp/miniconda3/envs/Wan2.2` (`Wan2.2`)
- Model weights: `/hy-tmp/models/Wan2.2-T2V-A14B`
- Model directory size: about `118G`
- OpenVid-100 prompts: `/hy-tmp/openvid_100_wan22_prompts.zip`
- Reports directory: `reports/`
- Result symlink directory: `experiment_results/`
- Data disk: `/hy-tmp`, about `400G`; current check showed about `136G` free.
- Current GPU check: `NVIDIA A100 80GB PCIe`, `81920 MiB`, driver `570.211.01`; GPU memory was idle at the latest check.
- Current tmux check: no tmux server was running at the latest check.
- OSS CLI was previously verified at `/usr/local/bin/oss`; relevant commands are `oss login`, `oss ls -s -d oss://datasets/`, and `oss cp ...`.
- The OpenVid first-50 handoff archive was previously created and uploaded as `oss://datasets/wan22_openvid_first50_handoff.tar.gz`; local source was `/hy-tmp/wan22_openvid_first50_handoff.tar.gz` with SHA256 `ee3458b05944e4fa5439f62e3a2896d9f9920dbd4beabc0938a86fd64dfe7b9e`.

## Implementation State

Primary code paths:

- CLI: `generate.py`
- T2V pipeline: `wan/text2video.py`
- timestep cache: `wan/timestep_cache.py`
- block cache: `wan/block_cache.py`, `wan/block_group_cache.py`
- CFG cache: `wan/cfg_cache.py`
- transformer integration: `wan/modules/model.py`

Implemented/available cache methods:

- `--timestep_cache zeus`
  - fixed ZEUS-style timestep cache.
- `--timestep_cache zeus-threshold`
  - threshold-based timestep cache using latent relative-L1.
  - recommended unified threshold alias: `--timestep_threshold`
  - historical alias still works: `--zeus_threshold`
- `--timestep_cache seacache`
  - SeaCache-style timestep/block-residual cache, used mainly as a comparison and data-generation method.
  - threshold argument: `--seacache_threshold`
- `--block_cache block-group`
  - recommended threshold-based block cache path.
  - recommended unified threshold alias: `--block_threshold`
  - historical alias still works: `--block_group_threshold`
- `--block_cache bwcache`
  - BWCache-style block cache.
  - historical argument `--bwcache_thresh`; compatibility alias `--bwcache_threshold`.
- `--cfg_cache threshold`
  - threshold-based CFG delta cache.
  - threshold argument: `--cfg_threshold`
- `--cfg_cache sea-threshold`
  - SeaCache-aligned CFG delta cache for cfg-cache-only experiments.
  - Uses the same first-block modulated norm feature shape as SeaCache, applies a scheduler-aware SEA frequency filter, and gates reuse with accumulated relative-L1.
  - Only first-step and tail-step protection are used by default (`--cfg_ret_steps 1`, `--cfg_cutoff_steps 1`); the old `--cfg_start/--cfg_end` window and `--cfg_max_reuse` cadence guard are not used by this mode.
  - Tunable SEA filter parameters: `--cfg_sea_power_exp`, `--cfg_sea_power_const`, `--cfg_sea_norm_mode`.

Important composition rule:

- CFG cache is the outer branch-selection cache.
- If CFG hits, skip uncond and reconstruct from cached CFG delta.
- If CFG misses, run uncond and refresh CFG delta.
- For each actual branch (`cond` or `uncond`), check timestep cache first.
- Only when timestep cache misses should block cache logic run.
- Only when both timestep and block caches miss should transformer blocks actually execute.
- Cache state must be keyed explicitly by `model_stage` and `branch`; never infer branch from call parity.

Recent interface cleanup:

- `generate.py` now exposes compatible unified aliases:
  - `--timestep_threshold` -> `--zeus_threshold`
  - `--block_threshold` -> `--block_group_threshold`
  - `--bwcache_threshold` -> `--bwcache_thresh`
- No cache implementation logic was changed during this cleanup.

## Experiment Defaults

Unless a task says otherwise, use these defaults for Wan2.2 T2V cache experiments:

- task/model: `t2v-A14B`
- checkpoint: `/hy-tmp/models/Wan2.2-T2V-A14B`
- seed: fixed `42`
- size: `832*480`
- frame count: `45`
- sampling steps: `50`
- solver: `dpm++`
- offload: `--offload_model`
- dtype conversion: `--convert_model_dtype`
- baseline: no-cache output with same prompt/seed/shape
- timing metric: `inference_compute_elapsed_seconds`
- quality metric: FFmpeg PSNR against baseline, excluding perfect/Infinity frames where applicable
- use single-process batch runners for threshold sweeps so one process loads the pipeline once and then runs all candidates.

## Important Experiment Results

Fixed ZEUS 10-prompt formal run:

- Result root: `/hy-tmp/wan22_zeus_timestep_cache_50step_45f_480p_full_20260608_114307`
- Overall speedup: `1.986x`
- Mean FFmpeg PSNR: `23.705 dB`
- No failures were recorded.

ZEUS-threshold 10-prompt reuse_interp run:

- Result root: `/hy-tmp/wan22_zeus_threshold_reuse_interp_10prompt_5th_20260608_195427`
- It is the main reference baseline/result root reused by later prompt-01/prompt-02 comparisons.
- Prompt-01 pilot showed threshold `0.005` had high PSNR but modest speedup; threshold `0.02+` dropped to roughly `18.6-18.9 dB` while giving higher speed.

Three-cache prompt-01 64-combination grid:

- Result root: `/hy-tmp/wan22_three_cache_threshold_grid_prompt01_50step_45f_480p_20260610_012518`
- Completed `64/64` candidates with no failed records.
- Fastest candidate: `ts_0p6__bg_1__cfg_1`, `4.080x`, PSNR `15.225 dB`.
- Best finite high-PSNR candidate above `25 dB`: `ts_0p005__bg_0p001__cfg_0p001`, `1.039x`, PSNR `26.954 dB`.
- Best speed with PSNR `>=22 dB`: `ts_0p005__bg_0p015__cfg_0p03`, `1.204x`, PSNR `23.448 dB`.
- Best speed with PSNR `>=20 dB`: `ts_0p005__bg_0p03__cfg_0p02`, `1.369x`, PSNR `20.042 dB`.

Cache ablation prompt-01:

- Result root: `/hy-tmp/wan22_cache_ablation_prompt01_50step_45f_480p_20260609_184625`
- Baseline compute time: `522.603s`
- `timestep_only`: `1.600x`, PSNR `18.606 dB`
- `block_only`: `1.362x`, PSNR `19.396 dB`
- `cfg_only`: `1.148x`, PSNR `21.571 dB`
- `timestep_block`: `1.748x`, PSNR `18.159 dB`
- `timestep_cfg`: `1.332x`, PSNR `20.910 dB`
- `block_cfg`: `1.352x`, PSNR `19.446 dB`
- `all_three`: `1.370x`, PSNR `19.603 dB`

SeaCache prompt-01:

- Result root: `/hy-tmp/wan22_seacache_50step_45f_480p_20260611_191733`
- Threshold `0.10`: `1.112x`, PSNR `36.303 dB`
- Threshold `0.20`: `1.569x`, PSNR `24.558 dB`
- Threshold `0.30`: `1.966x`, PSNR `20.562 dB`
- Threshold `0.50`: `2.779x`, PSNR `19.460 dB`

SeaCache prompt-02 dense/high-threshold sweeps:

- Dense root: `/hy-tmp/wan22_seacache_prompt02_dense_20260611_204826`
- High-threshold root: `/hy-tmp/wan22_seacache_prompt02_highthr_20260612_000218`
- Threshold `0.10`: `1.090x`, PSNR `45.532 dB`
- Threshold `0.20`: `1.562x`, PSNR `30.097 dB`
- Threshold `0.30`: `1.965x`, PSNR `29.582 dB`
- Threshold `0.50`: `2.641x`, PSNR `23.725 dB`
- Threshold `0.60`: `3.098x`, PSNR `20.262 dB`
- Threshold `0.80`: `3.499x`, PSNR `18.631 dB`
- Takeaway from prompt 01 and 02: SeaCache dominated ZEUS-threshold on the observed quality/speed frontier, especially on prompt 02.

OpenVid SeaCache local shard:

- Current local assignment: OpenVid prompts 76-100, zero-based `prompt_start=75`, `prompt_limit=25`.
- Thresholds: `0.10 0.15 0.20 0.25 0.30 0.40 0.50 0.60 0.70 0.80`.
- Result root launched previously: `/hy-tmp/wan22_seacache_openvid100_50step_45f_480p_20260612_002814`
- Workspace symlink: `experiment_results/wan22_seacache_openvid100_50step_45f_480p_20260612_002814`
- Verified complete on 2026-06-13: `25/25` baselines and `250/250` SeaCache candidates exist; `failed/` is empty; `runner.log` ends with `Completed experiment`.
- Result tables:
  - `/hy-tmp/wan22_seacache_openvid100_50step_45f_480p_20260612_002814/results/summary.csv`
  - `/hy-tmp/wan22_seacache_openvid100_50step_45f_480p_20260612_002814/results/aggregate_by_threshold.csv`
- Aggregate threshold results:
  - `0.10`: `1.113x`, mean PSNR `42.333 dB`, min PSNR `34.26 dB`
  - `0.15`: `1.412x`, mean PSNR `34.222 dB`, min PSNR `23.62 dB`
  - `0.20`: `1.575x`, mean PSNR `30.188 dB`, min PSNR `19.50 dB`
  - `0.25`: `1.844x`, mean PSNR `26.787 dB`, min PSNR `19.36 dB`
  - `0.30`: `1.976x`, mean PSNR `25.170 dB`, min PSNR `17.67 dB`
  - `0.40`: `2.418x`, mean PSNR `22.836 dB`, min PSNR `14.50 dB`
  - `0.50`: `2.746x`, mean PSNR `21.429 dB`, min PSNR `14.02 dB`
  - `0.60`: `3.112x`, mean PSNR `19.567 dB`, min PSNR `13.01 dB`
  - `0.70`: `3.337x`, mean PSNR `19.282 dB`, min PSNR `13.05 dB`
  - `0.80`: `3.517x`, mean PSNR `19.004 dB`, min PSNR `13.24 dB`

## Reports

Important report files are under `reports/`:

- `reports/report.md`
- `reports/report_main_experiments.md`
- `reports/report_supplementary_experiments.md`
- `reports/report_seacache_vs_zeus_threshold_prompt12.md`

The main report covers fixed ZEUS, ZEUS-threshold reuse_interp, and the three-cache grid. The supplementary report covers smoke tests, pilots, block-cache-only comparisons, ablations, and failed/superseded runs.

## Known Issues And Lessons

- Compute speedup should use `inference_compute_elapsed_seconds`, not full process wall time.
- FFmpeg/ffprobe should use the conda env path when tmux PATH is uncertain.
- True no-offload is not appropriate on this single A100 80GB setup without model-parallel changes; high and low DiT checkpoints plus T5 and activations are too large.
- BWCache originally had OOM risk because high-stage block state survived into low stage; block cache stage clearing and summary archiving were added.
- `cfg_force_uncond_recompute_on_miss` exists for explicit comparison, but current threshold-combination runs should normally leave it disabled.
- Reuse/recompute summaries should distinguish unique timestep counts from summed branch-call counts.
- OpenVid prompts are longer and more caption-like than the original 10 prompts; use stable `sample_id` filenames for dataset rows.

## Next Recommended Work

1. Build or update a consolidated threshold dataset table from completed result roots under `/hy-tmp/wan22_*` and `experiment_results/`.
2. Use the consolidated table to define the first adaptive-threshold predictor baseline.
3. Keep future progress entries concise and append-only from this reset point.

Note added 2026-06-29: this "Next Recommended Work" block is stale for adaptive-threshold work. A first adaptive threshold predictor and adaptive SeaCache inference prototype already exist in `adaptive_threshold_predictor/` and `adaptive_seacache_wan22/`, with reports under `reports/report_adaptive_predictor.md`, `reports/report_adaptive_predictor_training_curves.md`, and `reports/report_adaptive_seacache_train15_test5_and_overhead.md`. Current adaptive-related follow-up should be based on the newer `todo.md` item: the existing predictor performs poorly and needs VBench10 retesting plus diagnosis, not a from-scratch first predictor.

## 2026-06-29 Transformer Predictor Design

- Discussed replacing the current pooled-feature MLP adaptive threshold predictor with a lightweight DiT-style Transformer predictor.
- Decided to use a CLS readout design and not exactly reproduce Wan2.2 3D RoPE in the first version.
- Wrote architecture/hyperparameter proposal: `reports/report_transformer_predictor_architecture.md`.

## 2026-06-30 Adaptive Predictor Architecture/Result Review

- Reviewed newly added `adaptive_threshold_predictor` MiniDiT-CLS Transformer and 5-feature gated MLP against `reports/report_transformer_predictor_architecture.md` and `reports/report_gated_multifeature_mlp_architecture.md`.
- Architecture match is mostly correct: MiniDiT uses raw latent `[16,12,60,104]` with learned `Conv3d` patch `(3,12,8)`, 260 tokens, CLS readout, factorized learned 3D position embeddings, 2 AdaLN-modulated Transformer blocks, and output range `[0.10,0.80]`; gated MLP uses five 128-d pooled features, separate encoders, condition-dependent softmax gates, fused feature+condition head, and 83,526 params.
- Important caveat found: MiniDiT zero-initialized AdaLN gates mean the first backward pass gives zero gradient to patch embedding / attention / block MLP / condition embedding; only CLS/head and modulation learn initially. Checkpoints show those modules do later change, so it is not a fatal bug, but `--dit_gate_init` should be ablated against a small nonzero value.
- Training settings are broadly reasonable, but row-split metrics are optimistic because train/val contain all 100 sample IDs. Use sample split as the generalization signal for model selection and online VBench10 evaluation.
- Result anomalies: first MiniDiT run `wan22_adaptive_threshold_mini_dit_cls_convpatch_3x12x8_d96_l2_20260629_214241` has no `metrics.json`/checkpoint and appears incomplete; do not include it in comparisons. MiniDiT sample split early-stopped with best val MAE about `0.1145`; MiniDiT row split reached about `0.0380`; gated MLP sample split best val MAE about `0.1143`, row split 30 about `0.0757`, row split 100 about `0.0601`.

## 2026-06-30 Adaptive Predictor Architecture Diagrams

- Created SVG architecture diagrams for the two current adaptive threshold predictor designs:
  - `reports/assets/gated_multifeature_mlp_architecture.svg`
  - `reports/assets/mini_dit_cls_predictor_architecture.svg`
- Added a small reproducible generator script: `reports/make_adaptive_architecture_diagrams.py`.
- Inserted the SVG references near the top of `reports/report_gated_multifeature_mlp_architecture.md` and `reports/report_transformer_predictor_architecture.md`.
- Revised the diagrams after review: removed non-architecture note/metadata boxes, and redrew the 5-feature MLP to explicitly show `condition -> gate head + softmax -> g1..g5 -> gated feature fusion`.
- Revised the 5-feature diagram again after the gated MLP output head was range-mapped: output now shows `[0.10,0.80]`, the encoded-feature aggregation box covers all five branches, and right-side boxes were widened/shortened to avoid text overflow.

## 2026-06-30 Adaptive SeaCache MiniDiT Online Smoke

- Integrated the trained `mini_dit_cls` Transformer predictor into `adaptive_seacache_wan22` while keeping the legacy cached-feature MLP path compatible.
- `adaptive_seacache_wan22/cache.py` now auto-detects checkpoint type, supports `best_model_checkpoint.pt` payloads, and instantiates `MiniDiTCLSAdaptiveThresholdPredictor` from checkpoint/config metadata.
- `adaptive_seacache_wan22/generate_t2v.py` now exposes `--adaptive_model_type auto|mlp|mini_dit_cls` plus optional MiniDiT shape/hyperparameter overrides, and clears stale logging handlers so Wan summary logs are emitted.
- Ran a full T2V smoke on GPU with VBench10 prompt `vbench10_001` / prompt text `A woman is playing football.`, seed `42`, `832*480`, `45` frames, `50` steps, `dpm++`, target PSNR `25`, checkpoint `/hy-tmp/wan22_adaptive_threshold_mini_dit_cls_convpatch_rowsplit_packed_d96_l2_bs128_20260629_232659/best_model_checkpoint.pt`.
- Result root: `/hy-tmp/wan22_adaptive_seacache_mini_dit_vbench01_smoke_20260630_021304`.
- Output validated by ffprobe: `832x480`, `45` frames, `16 fps`, duration `2.8125s`.
- Metrics against matching dpm++ no-cache baseline `/hy-tmp/wan22_zeus_vbench10_50step_45f_480p_20260624_003030/baseline/vbench10_001.mp4`: compute elapsed `374.941s`, baseline compute `538.211s`, speedup `1.435x`, FFmpeg PSNR average `19.493 dB`.
- Cache summary: total reuse branch calls `42`, total recompute branch calls `58`; predicted thresholds ranged `0.1885-0.2852` with mean approximately `0.2068`.
- Result tables written to `/hy-tmp/wan22_adaptive_seacache_mini_dit_vbench01_smoke_20260630_021304/results/summary.csv` and `cache_key_summary.csv`.
- Note: this target-25 MiniDiT smoke undershot the requested quality target substantially on this prompt. Next useful step is a small target sweep or replay comparison against fixed SeaCache thresholds on the same prompt before scaling to VBench10.

## 2026-06-30 MiniDiT Split Compare 24-Candidate Pilot

- User requested a small online comparison of MiniDiT predictors trained with normal sample split vs row split:
  - datasets: VBench10 and OpenVid100 train distribution
  - target PSNRs: `22`, `28`
  - prompts per dataset: `3`
  - total candidates: `2 models * 2 targets * 2 datasets * 3 prompts = 24`
- Added runner:
  - `experiments/adaptive_seacache_mini_dit_split_compare_50step_45f_480p/run_batch.py`
  - `experiments/adaptive_seacache_mini_dit_split_compare_50step_45f_480p/run_tmux.sh`
- Runner reuses existing baselines only; it does not generate baselines.
  - VBench10 dpm++ baselines are reused from `/hy-tmp/wan22_zeus_vbench10_50step_45f_480p_20260624_003030`.
  - OpenVid100 train baselines are reused from `/hy-tmp/openvid_100_seacache_trace_data/...`; selected prompts are sample-split train IDs with existing baseline artifacts.
- Selected records:
  - VBench10: `vbench10_001`, `vbench10_002`, `vbench10_003`
  - OpenVid train: `openvid_002/openvidhd_part1_001`, `openvid_004/openvidhd_part1_003`, `openvid_005/openvidhd_part1_004`
- Compared checkpoints:
  - sample split: `/hy-tmp/wan22_adaptive_threshold_mini_dit_cls_convpatch_3x12x8_d96_l2_bs128_20260629_214906/best_model_checkpoint.pt`
  - row split: `/hy-tmp/wan22_adaptive_threshold_mini_dit_cls_convpatch_rowsplit_packed_d96_l2_bs128_20260629_232659/best_model_checkpoint.pt`
- Runner memory hygiene:
  - loads WanT2V pipeline once
  - creates a fresh adaptive SeaCache factory per candidate
  - writes summary/trace immediately
  - calls `clear_last_instance()`, restores `wan.text2video.SeaCacheTimestepCache`, deletes the factory, and calls `torch.cuda.empty_cache()` after each candidate
- CPU validation passed and found exactly 24 expected candidates.
- Launched tmux run:
  - tmux session: `wan22_adaptive_mini_dit_split_20260630_025328`
  - result root: `/hy-tmp/wan22_adaptive_seacache_mini_dit_split_compare_50step_45f_480p_20260630_025328`
  - runner log: `/hy-tmp/wan22_adaptive_seacache_mini_dit_split_compare_50step_45f_480p_20260630_025328/logs/runner.log`
- First candidate completed successfully:
  - `vbench10_001`, sample-split MiniDiT, target `22`
  - compute elapsed `283.984s`
  - baseline compute `538.211s`
  - speedup `1.895x`
  - FFmpeg PSNR `14.928 dB`
  - trace rows `100`, reuse decisions `50`, recompute decisions `50`
  - threshold mean `0.2725`
- The run was still active at the time of this progress update; monitor with `tmux attach -t wan22_adaptive_mini_dit_split_20260630_025328`.

## 2026-06-29 Raw Latent Packed Cache

- Built a full packed raw-latent cache for adaptive-threshold training without stopping the MiniDiT training run.
- Cache root: `/hy-tmp/wan22_adaptive_threshold_raw_latent_packed_cache_candidate_inverse_fp16_20260629_221805`
- Source trace root: `/hy-tmp/openvid_100_seacache_trace_data`
- Builder script: `adaptive_threshold_predictor/build_raw_latent_cache.py`
- Launch/log:
  - `/hy-tmp/wan22_adaptive_threshold_raw_latent_packed_cache_candidate_inverse_fp16_20260629_221805/commands/build_cache.sh`
  - `/hy-tmp/wan22_adaptive_threshold_raw_latent_packed_cache_candidate_inverse_fp16_20260629_221805/logs/build.log`
- Build config:
  - dataset mode `candidate_inverse`
  - dtype `float16`
  - latent shape `[16, 12, 60, 104]`
  - shard size `512`
  - batch size `16`
  - workers `2`
  - low IO/CPU priority via `ionice -c2 -n7 nice -n 10`
- Completion:
  - processed `50000/50000` examples
  - elapsed `650.864s`
  - throughput about `77 examples/s` by the end
  - `98` shard files
  - cache root size about `112G`
  - `/hy-tmp` had about `160G` free after completion
- Integrity checks:
  - `manifest.json` reports `num_examples=50000`, `num_shards=98`, `dtype=float16`, `latent_shape=[16, 12, 60, 104]`.

  - `metadata.pt` has 50k entries for `sample_id`, `timestep`, `target_psnr`, `threshold`, `step_index`, `source_index`, `shard_name`, and `shard_offset`.
  - First shard tensor shape is `[512, 16, 12, 60, 104]`; final shard tensor shape is `[336, 16, 12, 60, 104]`.

## 2026-06-29 Row Split Packed MiniDiT Training

- Added packed raw latent dataset support and row-level split support for diagnostic MiniDiT training:
  - `adaptive_threshold_predictor/data.py`
  - `adaptive_threshold_predictor/train_gate.py`
- New training arguments:
  - `--packed_latent_cache_dir`
  - `--preload_packed_latents`
  - `--split_mode {sample,row}`
- Validation:
  - `python -m py_compile adaptive_threshold_predictor/data.py adaptive_threshold_predictor/train_gate.py` passed.
  - CPU smoke with packed cache, row split, and `--max_examples 64` completed.
- Formal run launched:
  - tmux session: `wan22_mini_dit_rowsplit_packed_20260629_232659`
  - output root: `/hy-tmp/wan22_adaptive_threshold_mini_dit_cls_convpatch_rowsplit_packed_d96_l2_bs128_20260629_232659`
  - symlink: `experiment_results/wan22_adaptive_threshold_mini_dit_cls_convpatch_rowsplit_packed_d96_l2_bs128_20260629_232659`
  - launch script: `/hy-tmp/wan22_adaptive_threshold_mini_dit_cls_convpatch_rowsplit_packed_d96_l2_bs128_20260629_232659/commands/launch_train.sh`
  - log: `/hy-tmp/wan22_adaptive_threshold_mini_dit_cls_convpatch_rowsplit_packed_d96_l2_bs128_20260629_232659/logs/train.log`
- First epoch completed:
  - `val_mae=0.10507791430577636`
  - earlier sample-split Conv3d MiniDiT best was `0.1144591414630413`
  - high-threshold rows remain difficult; epoch-1 `threshold_0.80` val MAE was about `0.236`.
- Completed at 2026-06-30 00:49 CST:
  - all `30` epochs completed; no early stop triggered.
  - best epoch: `29`
  - best val MAE: `0.03800193872973323`
  - final epoch val MAE: `0.03809734165892005`
  - best epoch train MAE: `0.04076685686819256`
  - best epoch `threshold_0.70` val MAE: `0.03862547485851774`
  - best epoch `threshold_0.80` val MAE: `0.07903741441145846`
  - model parameters: `724513`
  - tmux exited normally; GPU memory returned to idle.
- Interpretation:
  - Row split is much easier than sample split; validation MAE improved from `0.1144591414630413` to `0.03800193872973323`.
  - This indicates the Conv3d MiniDiT predictor can learn same-video / row-level interpolation from the packed raw latents.
  - The earlier sample-split failure is more consistent with cross-sample generalization difficulty and/or `candidate_inverse` task mismatch than with the architecture being unable to fit the signal.
- Recommended first configuration:
  - `MiniDiTCLSAdaptiveThresholdPredictor`
  - latent patch size `(3, 12, 8)`, token grid `[4, 5, 13]`, `260` latent tokens plus CLS
  - `d_model=96`, `2` layers, `4` heads, `mlp_ratio=2.0`, dropout `0.05`
  - factorized learned 3D positional embeddings
  - DiT-style AdaLN conditioning from step fraction and target PSNR
  - output constrained to threshold range `[0.10, 0.80]`
- Rationale: current training set has about 40k step-level examples but much lower effective diversity, so the first Transformer should remain under roughly 1M parameters.

## 2026-06-29 Multi-Feature MLP Extension

- Read `adaptive_threshold_predictor/` to locate the legacy MLP data/model path:
  - raw trace dataset: `TraceStepThresholdDataset`
  - cached pooled feature dataset: `CachedFeatureThresholdDataset`
  - legacy MLP model: `ImprovedAdaCacheGate` / `CachedFeatureAdaCacheGate`
  - training entry: `adaptive_threshold_predictor/train_gate.py`
- Added backward-compatible multi-feature MLP support:
  - existing `--feature_set <name>` single-feature behavior remains valid.
  - new `--feature_sets <name> [<name> ...]` concatenates multiple latent-derived features.
  - cached path loads multiple `features_<feature>.pt` files from `--cache_dir`, validates row counts, and concatenates along feature dimension.
  - raw-latent path extracts each selected feature internally, concatenates pooled vectors, and projects them back to `hidden_dim` before the unchanged prediction head.
  - config/metrics/checkpoint feature-extractor metadata now records `feature_sets`.
- Updated README with multi-feature MLP usage.
- Validation:
  - `python -m py_compile adaptive_threshold_predictor/data.py adaptive_threshold_predictor/models.py adaptive_threshold_predictor/train_gate.py` passed.
  - single-feature cached CPU smoke passed with `--feature_set latent_pool`, `max_examples=256`, `epochs=1`.
  - multi-feature cached CPU smoke passed with `--feature_sets latent_pool temporal_var frame_diff_mean`, `max_examples=256`, `epochs=1`.
  - raw-latent random tensor forward passed for `feature_sets=("latent_pool", "temporal_var", "frame_diff_mean")`, output shape `[2, 1]`.
- No full GPU training run was launched in this session.

## 2026-06-29 Gated Multi-Feature MLP

- User noted that direct feature concatenation is a weak fusion design and requested Scheme B: Per-Feature MLP + Gated Fusion.
- Implemented a gated fusion MLP path while preserving the direct-concat multi-feature baseline:
  - CLI flag: `--feature_fusion concat|gated`
  - `concat` keeps the previous direct concatenation behavior.
  - `gated` uses four default features unless `--feature_sets` is explicitly provided:
    - `latent_pool`
    - `temporal_var`
    - `frame_diff_mean`
    - `frame_diff_var`
- Model additions in `adaptive_threshold_predictor/models.py`:
  - `DEFAULT_GATED_FEATURE_SETS`
  - `GatedFeatureFusionAdaCacheGate`
  - `CachedGatedFeatureAdaCacheGate`
  - `GatedMultiFeatureAdaCacheGate`
- Gated design:
  - each selected feature has its own MLP encoder.
  - timestep/target-PSNR condition embedding predicts a softmax gate over feature embeddings.
  - fused feature is the gate-weighted sum of per-feature embeddings.
  - prediction head consumes fused feature plus condition embedding.
- Data/training integration:
  - `CachedFeatureThresholdDataset` now keeps both the direct concatenated tensor (`batch["feature"]`) and per-feature tensors (`batch["features"][name]`).
  - `train_gate.py` routes cached gated runs through the per-feature dict.
  - raw-latent gated runs extract the same features inside the model.
  - `config.json` includes `selected_feature_sets` and `resolved_feature_embedding_dim`.
  - `val_predictions.csv` includes `gate_<feature>` columns for gated models.
- README updated with the recommended gated 4-feature MLP command.
- Validation:
  - `python -m py_compile adaptive_threshold_predictor/data.py adaptive_threshold_predictor/models.py adaptive_threshold_predictor/train_gate.py` passed.
  - raw-latent random forward passed with output shape `[2, 1]`, gate shape `[2, 4]`, and gate rows summing to `1`.
  - cached gated CPU smoke passed with 4 default features, `max_examples=256`, `epochs=1`.
  - cached gated CPU smoke with non-empty validation passed with `max_examples=1200`, `--save_val_predictions`; output CSV contains gate columns.
  - `git diff --check` passed.
- No full GPU training run was launched in this session.

## 2026-06-29 Gated Multi-Feature MLP No-Concat Revision

- User requested three follow-ups:
  - do not retain the multi-feature concat method.
  - calculate the current parameter count.
  - write an architecture report modeled after `reports/report_transformer_predictor_architecture.md`.
- Removed the multi-feature direct-concat path from the training interface:
  - removed `--feature_fusion concat|gated`.
  - multi-feature runs are now selected by passing `--feature_sets ...` with more than one feature, and they always use gated fusion.
  - single-feature legacy MLP remains available through `--feature_set <name>` for old ablations.
  - `ImprovedAdaCacheGate` was restored to single-feature behavior only.
  - `CachedFeatureThresholdDataset` no longer constructs a concatenated tensor for multi-feature datasets; it returns per-feature tensors for gated fusion.
- Recommended 4-feature gated run remains:
  - `--feature_sets latent_pool temporal_var frame_diff_mean frame_diff_var`
- Current recommended gated MLP parameter count:
  - hidden dim `64`
  - feature embedding dim `64`
  - per-feature input dim `128`
  - four features
  - total trainable parameters: `71,045`
  - breakdown: feature encoders `49,664`, condition encoder `4,352`, gate head `4,420`, prediction head `12,609`.
- Added architecture report:
  - `reports/report_gated_multifeature_mlp_architecture.md`
- Updated README to remove the concat multi-feature command and document gated fusion as the only multi-feature MLP path.
- Validation:
  - `python -m py_compile adaptive_threshold_predictor/data.py adaptive_threshold_predictor/models.py adaptive_threshold_predictor/train_gate.py` passed.
  - cached gated CPU smoke passed with `--feature_sets latent_pool temporal_var frame_diff_mean frame_diff_var`, `max_examples=64`, `epochs=1`.
  - cached gated CPU smoke with non-empty validation passed with `max_examples=1200`, `--save_val_predictions`; `val_predictions.csv` contains the four gate columns.
- No full GPU training run was launched in this revision.

## 2026-06-30 Gated MLP Sample-Split Training

- User requested a normal train/test run using sample-level train/validation split.
- Launched and completed gated 4-feature MLP training on GPU before GPU was turned off.
- Result root:
  - `/hy-tmp/wan22_adaptive_threshold_mlp_gated_4feature_samplesplit_20260630_013006`
- Workspace symlink:
  - `experiment_results/wan22_adaptive_threshold_mlp_gated_4feature_samplesplit_20260630_013006`
- Launch script:
  - `/hy-tmp/wan22_adaptive_threshold_mlp_gated_4feature_samplesplit_20260630_013006/commands/launch_train.sh`
- Log:
  - `/hy-tmp/wan22_adaptive_threshold_mlp_gated_4feature_samplesplit_20260630_013006/logs/train.log`
- Training configuration:
  - `--model_type mlp`
  - `--cache_dir /hy-tmp/wan22_adaptive_threshold_feature_cache_candidate_inverse_20260616_012409`
  - `--feature_sets latent_pool temporal_var frame_diff_mean frame_diff_var`
  - `--dataset_mode candidate_inverse`
  - `--split_mode sample`
  - `--epochs 30`
  - `--batch_size 256`
  - `--hidden_dim 64`
  - `--feature_embedding_dim 64`
  - `--lr 3e-4`
  - `--min_lr 1e-5`
  - `--warmup_steps 500`
  - `--weight_decay 1e-4`
  - `--smooth_l1_beta 0.02`
  - `--grad_clip 1.0`
  - `--early_stop_patience 5`
  - `--dit_dropout 0.05`
  - `--save_val_predictions`
- Split:
  - train examples: `40000`
  - validation examples: `10000`
  - train samples: `80`
  - validation samples: `20`
- Model parameters:
  - `71045`
- Completion:
  - ran `13` epochs
  - early stopped at epoch `13`
  - best epoch: `8`
  - best validation MAE: `0.11151796581298112`
  - best validation loss: `0.10222012972831726`
  - best epoch train MAE: `0.08612967762444168`
  - final epoch validation MAE: `0.11888088526204228`
- Best epoch validation summary:
  - bias: `-0.045118419414758686`
  - prediction min/max/mean/std: `0.07821400463581085` / `0.7639604210853577` / `0.3548815858006477` / `0.18562251329421997`
  - MAE by step:
    - `step_00_09`: `0.11781573643535376`
    - `step_10_39`: `0.10946383323520421`
    - `step_40_49`: `0.11138259292393923`
  - MAE by threshold:
    - `0.10`: `0.015568429000675678`
    - `0.15`: `0.03470603171736002`
    - `0.20`: `0.05534320007264614`
    - `0.25`: `0.09033234791457653`
    - `0.30`: `0.1060278910547495`
    - `0.40`: `0.11715792307257653`
    - `0.50`: `0.12313540376722813`
    - `0.60`: `0.12578511033952236`
    - `0.70`: `0.18056872788071632`
    - `0.80`: `0.2665545933097601`
- Final checkpoint validation prediction gate means:
  - `gate_latent_pool`: `0.6956643772833049`
  - `gate_temporal_var`: `0.09785542918057182`
  - `gate_frame_diff_mean`: `0.1012345067290822`
  - `gate_frame_diff_var`: `0.10524568697217619`
- Interpretation:
  - Sample-split gated MLP best val MAE `0.1115` is close to the previous sample-split MiniDiT best `0.1144591414630413`, while using far fewer parameters (`71k` vs `724k`).
  - Like MiniDiT sample split, high thresholds remain hardest; best epoch `threshold_0.80` val MAE is `0.2666`.
  - Gate weights on the final checkpoint are dominated by `latent_pool`, especially at higher thresholds; motion features receive more weight at low threshold `0.10` than at high thresholds.
  - This suggests the gated model is mostly using raw pooled latent features under the current `candidate_inverse` setup, with limited but nonzero contribution from motion-derived features.

## 2026-06-30 Gated MLP Row-Split Training

- User requested row-split training for the gated 4-feature MLP.
- Initial CPU attempt was started because GPU had been turned off, but it was too slow and produced no epoch output after several minutes.
- After GPU was restored, the CPU attempt residual was removed:
  - deleted `/hy-tmp/wan22_adaptive_threshold_mlp_gated_4feature_rowsplit_cpu_20260630_014051`
  - deleted `experiment_results/wan22_adaptive_threshold_mlp_gated_4feature_rowsplit_cpu_20260630_014051`
- Launched and completed GPU row-split training.
- Result root:
  - `/hy-tmp/wan22_adaptive_threshold_mlp_gated_4feature_rowsplit_gpu_20260630_014852`
- Workspace symlink:
  - `experiment_results/wan22_adaptive_threshold_mlp_gated_4feature_rowsplit_gpu_20260630_014852`
- Launch script:
  - `/hy-tmp/wan22_adaptive_threshold_mlp_gated_4feature_rowsplit_gpu_20260630_014852/commands/launch_train.sh`
- Log:
  - `/hy-tmp/wan22_adaptive_threshold_mlp_gated_4feature_rowsplit_gpu_20260630_014852/logs/train.log`
- Training configuration:
  - `--model_type mlp`
  - `--cache_dir /hy-tmp/wan22_adaptive_threshold_feature_cache_candidate_inverse_20260616_012409`
  - `--feature_sets latent_pool temporal_var frame_diff_mean frame_diff_var`
  - `--dataset_mode candidate_inverse`
  - `--split_mode row`
  - `--epochs 30`
  - `--batch_size 256`
  - `--hidden_dim 64`
  - `--feature_embedding_dim 64`
  - `--lr 3e-4`
  - `--min_lr 1e-5`
  - `--warmup_steps 500`
  - `--weight_decay 1e-4`
  - `--smooth_l1_beta 0.02`
  - `--grad_clip 1.0`
  - `--early_stop_patience 5`
  - `--dit_dropout 0.05`
  - `--device cuda`
  - `--save_val_predictions`
- Split:
  - train examples: `40000`
  - validation examples: `10000`
  - split mode: `row`
- Model parameters:
  - `71045`
- Completion:
  - ran all `30` epochs
  - no early stop triggered
  - best epoch: `30`
  - best validation MAE: `0.07593761396706104`
  - best validation loss: `0.06691185193061829`
  - best epoch train MAE: `0.07653351860381663`
- Best epoch validation summary:
  - bias: `-0.0032232597440481187`
  - prediction min/max/mean/std: `0.07465098053216934` / `0.8386164307594299` / `0.39608174546957015` / `0.20561251044273376`
  - MAE by step:
    - `step_00_09`: `0.0928498020344101`
    - `step_10_39`: `0.07261506491243311`
    - `step_40_49`: `0.0687483228470994`
  - MAE by threshold:
    - `0.10`: `0.014997302341942836`
    - `0.15`: `0.03539148792775295`
    - `0.20`: `0.045847805324606423`
    - `0.25`: `0.06859489056477255`
    - `0.30`: `0.07987874489255815`
    - `0.40`: `0.10076054226516223`
    - `0.50`: `0.0864502436272883`
    - `0.60`: `0.07383361327600999`
    - `0.70`: `0.09040507235947777`
    - `0.80`: `0.16622635118142076`
- Final checkpoint validation prediction gate means:
  - `gate_latent_pool`: `0.602346860973537`
  - `gate_temporal_var`: `0.1366311730541289`
  - `gate_frame_diff_mean`: `0.14120697866557166`
  - `gate_frame_diff_var`: `0.11981498754592612`
- Interpretation:
  - Row split is easier than sample split for this model: best val MAE improved from `0.11151796581298112` to `0.07593761396706104`.
  - Compared with row-split MiniDiT best val MAE `0.03800193872973323`, this 71k-param gated MLP is worse but still substantially better than sample split.
  - Final gate means are less dominated by `latent_pool` than sample split (`0.60` vs `0.70`), with more weight assigned to temporal/motion features.
  - As with other runs, the high threshold `0.80` remains the hardest bucket.
- GPU was idle again after completion.

## 2026-06-29 Transformer Predictor Implementation

- Implemented the first MiniDiT-CLS adaptive threshold predictor code.
- Added:
  - `MiniDiTCLSAdaptiveThresholdPredictor` in `adaptive_threshold_predictor/models.py`
  - `GridMLPThresholdPredictor` capacity baseline in `adaptive_threshold_predictor/models.py`
  - `GridFeatureThresholdDataset` and `collate_grid_features` in `adaptive_threshold_predictor/data.py`
  - grid cache builder `adaptive_threshold_predictor/build_grid_feature_cache.py`
  - `train_gate.py` support for `--model_type mini_dit_cls` and `--model_type grid_mlp`
- Training output now preserves richer analysis files:
  - `config.json`, `split.json`, `model_summary.json`
  - `epoch_metrics.jsonl`, `epoch_metrics.csv`, `metrics.json`
  - plain state dicts and metadata checkpoints for best/final models
  - optional final and per-epoch validation predictions
- Recommended commands were added to `adaptive_threshold_predictor/README.md`.
- Validation:
  - `python -m py_compile` passed for `models.py`, `data.py`, `build_grid_feature_cache.py`, and `train_gate.py`.
  - `git diff --check` passed for the modified adaptive predictor files.
  - CPU smoke test built a 16-example grid cache at `/hy-tmp/wan22_mini_dit_grid_cache_smoke_20260629`, producing grid shape `[16, 4, 5, 13]`.
  - CPU smoke tests ran one epoch for reduced-size `mini_dit_cls` and `grid_mlp` models and produced expected metrics/checkpoint/prediction files.
- No formal full-data training or VBench10 adaptive inference run was launched in this session.

## 2026-06-29 Transformer Predictor Conv3d Revision

- User requested that the predictor use the report's actual Conv3d patch embedding instead of the earlier avg-pooled grid prototype, and asked to ignore control-group work for now.
- Revised `MiniDiTCLSAdaptiveThresholdPredictor` to use raw latent input:
  - input `[B,16,12,60,104]`
  - `Conv3d(16, d_model, kernel_size=(3,12,8), stride=(3,12,8))`
  - token grid `[4,5,13]`, token count `260`
  - CLS readout and AdaLN-style conditioning retained
- Revised `train_gate.py` so `--model_type mini_dit_cls` uses `TraceStepThresholdDataset` directly and no longer requires `--grid_cache_dir`.
- Checkpoint metadata now includes `feature_extractor` with `type=learned_conv3d_patch_embedding`, `input_shape`, `patch_size`, `token_grid_shape`, and `token_count`.
- Changed shared training defaults to the recommended MiniDiT settings: batch size `64`, epochs `30`, lr `3e-4`, warmup `500`, SmoothL1 beta `0.02`, grad clip `1.0`, early-stop patience `5`.
- Updated `adaptive_threshold_predictor/README.md` and `reports/report_transformer_predictor_architecture.md` to document Conv3d patch embedding as the recommended `mini_dit_cls` path.
- Validation:
  - CPU smoke test ran `mini_dit_cls` from raw latent with `--dit_patch_size 3 12 8`.
  - Smoke checkpoint metadata confirmed `token_grid_shape=[4,5,13]` and `token_count=260`.

## 2026-06-29 MiniDiT Convpatch Training Launch

- Launched full-data MiniDiT-CLS Conv3d patch predictor training in tmux.
- Active tmux session:
  - `wan22_mini_dit_convpatch_train_20260629_214906`
- Result root:
  - `/hy-tmp/wan22_adaptive_threshold_mini_dit_cls_convpatch_3x12x8_d96_l2_bs128_20260629_214906`
- Workspace symlink:
  - `experiment_results/wan22_adaptive_threshold_mini_dit_cls_convpatch_3x12x8_d96_l2_bs128_20260629_214906`
- Launch script:
  - `/hy-tmp/wan22_adaptive_threshold_mini_dit_cls_convpatch_3x12x8_d96_l2_bs128_20260629_214906/commands/launch_train.sh`
- Log:
  - `/hy-tmp/wan22_adaptive_threshold_mini_dit_cls_convpatch_3x12x8_d96_l2_bs128_20260629_214906/logs/train.log`
- Training configuration:
  - `--model_type mini_dit_cls`
  - `--dataset_mode candidate_inverse`
  - `--batch_size 128`
  - `--epochs 30`
  - `--lr 3e-4`
  - `--min_lr 1e-5`
  - `--warmup_steps 500`
  - `--smooth_l1_beta 0.02`
  - `--grad_clip 1.0`
  - `--early_stop_patience 5`
  - `--dit_dim 96`
  - `--dit_layers 2`
  - `--dit_heads 4`
  - `--dit_patch_size 3 12 8`
  - `--num_workers 8`
  - `--save_val_predictions`
  - `--save_epoch_val_predictions`
- Pre-launch checks:
  - GPU available and idle: A100 80GB.
  - `/hy-tmp` had about `68G` free.
  - Raw latent DataLoader throughput check after suppressing repeated `torch.load` warnings: `512` examples with `num_workers=8` took about `8.833s` on that short test (`57.96 examples/s`; cold-cache/worker-startup affected).
  - GPU smoke with `batch_size=128` passed.
- Important fix before final launch:
  - First tmux launch at `20260629_214241` was stopped before epoch output because repeated PyTorch `torch.load` FutureWarnings were emitted for every raw latent load and slowed/noised the run.
  - `adaptive_threshold_predictor/data.py` was updated to use `weights_only=True` for trace/cache `torch.load` calls, eliminating the warning flood.
- First epoch result from active run:
  - epoch elapsed: `200.632s`
  - train loss: `0.1223346`
  - train MAE: `0.1319188`
  - val loss: `0.1085956`
  - val MAE: `0.1181465`
  - val prediction range: min `0.1064772`, max `0.7293686`, mean `0.4027086`, std `0.1986593`
- Status at latest check:
  - tmux session still running.
  - `epoch_metrics.jsonl` contains epoch 1.
  - `best_model.pt`, `best_model_checkpoint.pt`, and `val_predictions_epoch_001.csv` were created.

## 2026-06-29 Raw Latent Packed Cache Build

- User requested building a full raw latent packed cache under `/hy-tmp` without stopping the active MiniDiT training.
- Added cache builder:
  - `adaptive_threshold_predictor/build_raw_latent_cache.py`
- The builder packs raw step latents from `TraceStepThresholdDataset` into fp16 shard files:
  - one shard default: `512` examples
  - latent shape per example: `[16, 12, 60, 104]`
  - expected full cache size: roughly `115-120G`
- Result root:
  - `/hy-tmp/wan22_adaptive_threshold_raw_latent_packed_cache_candidate_inverse_fp16_20260629_221805`
- Active tmux session:
  - `wan22_raw_latent_cache_build_20260629_221805`
- Launch script:
  - `/hy-tmp/wan22_adaptive_threshold_raw_latent_packed_cache_candidate_inverse_fp16_20260629_221805/commands/build_cache.sh`
- Log:
  - `/hy-tmp/wan22_adaptive_threshold_raw_latent_packed_cache_candidate_inverse_fp16_20260629_221805/logs/build.log`
- Build command uses low priority:
  - `ionice -c2 -n7 nice -n 10`
  - `--dtype float16`
  - `--shard_size 512`
  - `--batch_size 16`
  - `--num_workers 2`
- Pre-launch checks:
  - `/hy-tmp` was expanded to `800G`, with about `268G` free.
  - Source trace root `/hy-tmp/openvid_100_seacache_trace_data` was about `135G`.
  - 20-example smoke cache succeeded.
- Early build progress:
  - `8192/50000` examples processed in `127.98s`
  - throughput about `64 examples/s`
  - `16` shard files written, cache directory size about `19G`
  - projected completion time about `13-15 minutes` at early throughput
- Active MiniDiT training was not stopped. At the same check it had reached epoch `9`.

## 2026-06-23 Project Compute Cost Scan

- Scanned experiment result roots under `/hy-tmp`, including `wan22_*` archives and `/hy-tmp/openvid_100_seacache_trace_data`.
- Cost rule requested by user: actual A100 occupancy is estimated as `inference compute elapsed * 1.5` to account for model loading, debug, save/eval scheduling, and other GPU holding overhead.
- Billing rate: single A100 at `6 CNY/hour`.
- Deduplication rule: use canonical `summary.csv` detail rows where available; exclude aggregate/interim/partial duplicate tables, handoff build copies, repo mirror copies, and VBench shard/per-prompt summaries when merged summaries exist. Baseline elapsed values repeated across rows are counted once per sample within each result table. Empty or incomplete summaries were supplemented only from `partial_summary.csv`, `summary_interim_prompt01.csv`, or logs containing `inference_compute_elapsed_seconds`.
- Result: structured/parseable inference compute time `222.267 GPU-hours`; adjusted occupancy `333.401 A100-hours`; estimated cost `2000.40 CNY`.

## 2026-06-24 VBench10 ZEUS / SeaCache Report

- Added `reports/report_vbench10_zeus_threshold_seacache.md`.

## 2026-06-29 Adaptive Predictor Architecture Review

- Reviewed the original MLP adaptive threshold predictor in `adaptive_threshold_predictor/`:
  - MLP path uses pooled latent-derived features plus timestep/PSNR condition MLP.
  - Default dataset mode remains `candidate_inverse`: candidate latent + achieved PSNR -> threshold label.
  - Historical training showed strong early overfitting; `2x2x2 temporal_mean`, especially `hidden_dim=16`, was the best lightweight MLP setting.
- Reviewed the new Transformer candidate implementation:
  - `build_grid_feature_cache.py` creates fixed `[16,4,5,13]` grid features with `avg_pool3d` patch `(3,12,8)`.
  - `MiniDiTCLSAdaptiveThresholdPredictor` uses factorized learned 3D position embeddings, CLS readout, AdaLN-style conditioning, and output constrained to `[0.10,0.80]`.
  - Current implementation is lighter than the architecture proposal because it uses average-pooled grid features plus `Linear(16->dim)`, not a learnable `Conv3d(16->dim,kernel=stride=(3,12,8))` patch embed.
- Main review conclusions:
  - Network shape and conditioning design are broadly reasonable as a small first Transformer, but it should be compared against `grid_mlp` and the old MLP because pooled grid features may be the main improvement, not attention.
  - The largest methodological risk remains data/label design: `candidate_inverse` trains on achieved PSNR and candidate-run latents, while online inference uses desired PSNR and adaptive-run latents.
  - Transformer training defaults in `train_gate.py` are still legacy defaults unless overridden; first real runs should explicitly use lr `3e-4`, batch size around `64`, `SmoothL1 beta=0.02`, grad clip `1.0`, and early stopping.
  - No code changes were made during this review beyond this progress entry and the session log.

## 2026-06-29 Adaptive Predictor Recheck After ConvPatch Revision

- Rechecked the user-updated Transformer predictor implementation.
- The `mini_dit_cls` path now uses raw traced latent inputs and a learnable `Conv3d(16 -> dim, kernel=stride=(3,12,8))` patch embedding, matching the architecture report more closely than the earlier avg-pooled grid prototype.
- `train_gate.py` now routes `mini_dit_cls` through `TraceStepThresholdDataset`, adds `--dit_patch_size`, uses Transformer-oriented default training settings, and stores `feature_extractor` metadata in metrics/checkpoints.
- Lightweight validation:
  - `py_compile` passed for adaptive predictor modules.
  - A reduced MiniDiT forward pass produced `[1,1]` output within `[0.10,0.80]`.
  - Default `dim=96,layers=2,heads=4` model has `724,513` trainable parameters and patch grid `(16,4,5,13)`, consistent with the report's scale.
- Remaining review notes:
  - The data/label mismatch remains: `candidate_inverse` still trains on achieved PSNR and fixed-threshold candidate latents, while online adaptive inference will use desired PSNR and adaptive-run latents.
  - Changing shared defaults in `train_gate.py` affects old MLP runs unless their commands explicitly override lr/epochs/batch/loss settings.
  - The fixed avg-pooled grid cache path remains useful for `grid_mlp` controls but is no longer the recommended `mini_dit_cls` path.
- Report covers only VBench10 experiment environment/settings, per-prompt results, and 10-prompt aggregate results for fixed ZEUS, ZEUS-threshold, and timestep-only SeaCache.
- Data sources:
  - `/hy-tmp/wan22_zeus_vbench10_50step_45f_480p_20260624_003030/results/summary.csv`
  - `/hy-tmp/wan22_zeus_vbench10_50step_45f_480p_20260624_003030/results/aggregate_by_method.csv`
  - `/hy-tmp/wan22_vbench10_three_cache_full_merge_and_timestep_only_full_20260623/timestep_only_seacache_vbench10_full/wan22_seacache_vbench10_50step_45f_480p_20260618_161845/merged/summary.csv`
  - `/hy-tmp/wan22_vbench10_three_cache_full_merge_and_timestep_only_full_20260623/timestep_only_seacache_vbench10_full/wan22_seacache_vbench10_50step_45f_480p_20260618_161845/merged/aggregate_by_threshold.csv`

## 2026-06-24 ZEUS Solver Comparison Reports

- Added `reports/report_zeus_dpmpp_ali10_20260624.md` from local fixed-ZEUS `dpm++` ali-10 formal run.
- Added `reports/report_zeus_dpmpp_vbench10_20260624.md` from local fixed-ZEUS `dpm++` VBench10 run.
- Added `reports/report_zeus_solver_ali10_vbench10_comparison_20260624.md`, comparing four reports: ali-10/VBench10 crossed with `dpm++`/`unipc`.
- Largest adjusted-cost components:
  - `/hy-tmp/openvid_100_seacache_trace_data`: `134.696 A100-hours`, `808.18 CNY`.
  - VBench three-cache merged run: `86.679 A100-hours`, `520.07 CNY`.
  - `/hy-tmp/wan22_seacache_openvid100_50step_45f_480p_20260612_002814`: `32.876 A100-hours`, `197.25 CNY`.

## 2026-06-25 Commit And Push

- Reviewed the pending report/log/progress changes and confirmed they are small Markdown handoff artifacts.
- Prepared to commit the VBench10 ZEUS/SeaCache report set, ZEUS solver comparison reports, session logs, and this progress update.

## 2026-06-18 Current Repo Remote Handoff

- Handoff workflow clarified: do not package the current repository into OSS. Keep code in GitHub; put only reusable runtime artifacts such as the packed conda environment in OSS.
- Reused existing packed environment from the earlier handoff:
  - Local source: `/hy-tmp/wan22_openvid_first50_handoff_build/Wan2.2/env/Wan2.2-conda-env.tar.gz`
  - SHA256: `348f63583d2a3ea742b80341dbb97043c6a497065e593a1329b1aad1a0551f03`
  - Uploaded OSS target: `oss://datasets/Wan2.2-conda-env.tar.gz`

## 2026-06-19 Adaptive SeaCache Train-Split OpenVid-10 Launch

- Added a dedicated runner under `experiments/adaptive_seacache_train10_50step_45f_480p/`.
- Purpose: run the same timestep-only adaptive SeaCache inference setting on 10 prompts randomly sampled from the adaptive predictor train split.
- Sampling:
  - Predictor split: `/hy-tmp/wan22_adaptive_threshold_feature_ablation_hdim16_20260616/temporal_mean/split.json`
  - Prompt source: `test_sets/openvid_100/prompts.jsonl`
  - Random seed: `20260619`
  - Selected source IDs: `openvidhd_part1_085`, `openvidhd_part1_086`, `openvidhd_part1_059`, `openvidhd_part1_057`, `openvidhd_part1_016`, `openvidhd_part1_036`, `openvidhd_part1_093`, `openvidhd_part1_063`, `openvidhd_part1_095`, `openvidhd_part1_058`
- Baselines are reused from `/hy-tmp/work/Wan2.2/experiment_results/openvid_100_seacache_trace_data`; this runner does not regenerate no-cache baselines.
- Candidate settings:
  - target PSNRs: `20 25 30`
  - seed: `42`
  - size: `832*480`
  - frames: `45`
  - steps: `50`
  - solver: `dpm++`
  - `--offload_model true`
  - `--convert_model_dtype`
  - predictor checkpoint: `/hy-tmp/wan22_adaptive_threshold_feature_ablation_hdim16_20260616/temporal_mean/best_model.pt`
- CPU validation passed: 10 selected train prompts and all reusable baseline videos/logs/time/ffprobe artifacts were found.
- Launched tmux session `adaptive_seacache_train10` at 2026-06-19 13:45 CST.
- Experiment root: `/hy-tmp/wan22_adaptive_seacache_train10_50step_45f_480p_20260619_134522`
- Initial log check showed the adaptive gate checkpoint loaded successfully and no immediate error.
- Superseded shortly after launch by the larger train15/test5 request. The tmux session `adaptive_seacache_train10` was stopped after only one candidate had completed. Treat this root as an interrupted pilot, not as the formal result set.

## 2026-06-19 Adaptive SeaCache Train15/Test5 Launch

- Added a dedicated runner under `experiments/adaptive_seacache_train15_test5_50step_45f_480p/`.
- Purpose: run the same timestep-only adaptive SeaCache inference setting on 20 prompts from the predictor split: 15 train prompts and 5 held-out validation/test prompts.
- Sampling:
  - Predictor split: `/hy-tmp/wan22_adaptive_threshold_feature_ablation_hdim16_20260616/temporal_mean/split.json`
  - Prompt source: `test_sets/openvid_100/prompts.jsonl`
  - Random seed: `20260619`
  - Train source IDs: `openvidhd_part1_085`, `openvidhd_part1_086`, `openvidhd_part1_059`, `openvidhd_part1_057`, `openvidhd_part1_016`, `openvidhd_part1_036`, `openvidhd_part1_093`, `openvidhd_part1_063`, `openvidhd_part1_095`, `openvidhd_part1_058`, `openvidhd_part1_027`, `openvidhd_part1_012`, `openvidhd_part1_020`, `openvidhd_part1_031`, `openvidhd_part1_037`
  - Test/val source IDs: `openvidhd_part1_028`, `openvidhd_part1_030`, `openvidhd_part1_026`, `openvidhd_part1_092`, `openvidhd_part1_055`
- Baselines are reused from `/hy-tmp/work/Wan2.2/experiment_results/openvid_100_seacache_trace_data`; this runner does not regenerate no-cache baselines.
  - Candidate settings:
  - target PSNRs: `20 25 30`

## 2026-06-23 Adaptive Predictor Training Curve Report

- Added training-curve plotting script:
  - `experiments/adaptive_threshold_predictor/plot_training_curves.py`
- Generated adaptive predictor figures and summaries:
  - `reports/assets/adaptive_training_curves/feature_curves_2x2x2.svg`
  - `reports/assets/adaptive_training_curves/grid_val_curves_by_feature.svg`
  - `reports/assets/adaptive_training_curves/best_val_loss_heatmap.svg`
  - `reports/assets/adaptive_training_curves/long_run_curves.svg`
  - `reports/assets/adaptive_training_curves/training_curve_summary.csv`
  - `reports/assets/adaptive_training_curves/training_curve_summary.json`
- Added report:
  - `reports/report_adaptive_predictor_training_curves.md`
- Main conclusion:
  - `2x2x2 temporal_mean` remains the best 3-epoch feature/grid setting, best validation loss `0.012259`.
  - Larger pooling grids do not improve global best validation loss.
  - no-feature/noise-feature controls are around `0.01465` validation loss, confirming that real latent-derived features provide useful signal.
  - 30-epoch curves show early validation optimum followed by overfitting; current experiments should be interpreted with early stopping rather than as fully converged fixed-epoch training.

## 2026-06-23 ZEUS Result Lookup Session

- Reviewed `wan/timestep_cache.py`, `wan/text2video.py`, and existing reports to answer how `zeus` / `zeus-threshold` reuse estimates are produced.
- Confirmed both methods reuse whole branch denoiser outputs; default `reuse_interp` alternates the first extrapolated `prev_interp` with the latest real recompute output during consecutive reuse runs.
- Looked up archived ZEUS results without launching new inference:
  - fixed ZEUS formal 10-prompt root: `/hy-tmp/wan22_zeus_timestep_cache_50step_45f_480p_full_20260608_114307`
  - ZEUS-threshold reuse_interp 10-prompt root: `/hy-tmp/wan22_zeus_threshold_reuse_interp_10prompt_5th_20260608_195427`
  - prompt-01 pilot root: `/hy-tmp/wan22_zeus_threshold_prompt01_7th_20260608_162827`
  - prompt-01 timestep-aware interpolation root: `/hy-tmp/wan22_zeus_threshold_taware_prompt01_5th_20260608_191714`
- No code, cache logic, experiment artifacts, or result tables were changed.

  - total candidates: `60`
  - seed: `42`
  - size: `832*480`
  - frames: `45`
  - steps: `50`
  - solver: `dpm++`
  - `--offload_model true`
  - `--convert_model_dtype`
  - predictor checkpoint: `/hy-tmp/wan22_adaptive_threshold_feature_ablation_hdim16_20260616/temporal_mean/best_model.pt`
- CPU validation passed: 20 selected prompts and all reusable baseline videos/logs/time/ffprobe artifacts were found.
- Launched tmux session `adaptive_seacache_train15_test5` at 2026-06-19 13:55 CST.
- Experiment root: `/hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521`
- Initial log check showed the adaptive gate checkpoint loaded successfully and no immediate error.

## 2026-06-24 ZEUS VBench10 Experiment Scripts

- Added `experiments/zeus_vbench10_50step_45f_480p/` to rerun fixed ZEUS and ZEUS-threshold on the unified `test_sets/Vbench10/prompts.jsonl` subset.
- New files:
  - `run_batch.py`: single-process WanT2V runner; loads the pipeline once, runs 10 no-cache baselines, 10 fixed-ZEUS candidates, and 50 ZEUS-threshold candidates by default.
  - `summarize_results.py`: writes one-row-per-candidate `results/summary.csv` and aggregate `results/aggregate_by_method.csv/json`.
  - `run_tmux.sh`: launches the full run in tmux and creates an `experiment_results/` symlink.
  - `README.md`: documents config, validation, launch, and outputs.
- Configuration intentionally matches earlier ZEUS experiments:
  - task `t2v-A14B`, checkpoint `/hy-tmp/models/Wan2.2-T2V-A14B`, seed `42`, size `832*480`, `45` frames, `50` DPM++ steps, `--offload_model`, `--convert_model_dtype`.
  - fixed ZEUS: acc range `8 <= step < 47`, denominator `3`, modular `0 1`, `reuse_interp`, max interval `6`, lagrange `4/4/24`.
  - ZEUS-threshold: thresholds `0.005 0.02 0.08 0.20 0.60`, `reuse_interp`, same acc range and max interval.
  - block cache and CFG cache disabled.
- Validation completed without launching inference:
  - `python -m py_compile experiments/zeus_vbench10_50step_45f_480p/run_batch.py experiments/zeus_vbench10_50step_45f_480p/summarize_results.py`
  - `python experiments/zeus_vbench10_50step_45f_480p/run_batch.py --cpu_validate`
  - `python experiments/zeus_vbench10_50step_45f_480p/run_batch.py --cpu_validate --prompt_limit 1 --thresholds '0.005 0.02'`
- Full GPU experiment was not launched in this script-preparation session.

## 2026-06-19 Adaptive SeaCache Predictor Overhead Setup

- Added predictor overhead instrumentation to `adaptive_seacache_wan22/cache.py`:
  - `AdaptiveSeaCacheGateConfig.measure_predictor_timing`
  - per-call `predictor_elapsed_seconds` in adaptive decision trace
  - summary fields for predictor total/mean/max/call count
  - `ReplaySeaCacheTimestepCache` and `build_replay_seacache_factory()` for threshold-trace replay without predictor calls
- Added overhead runner under `experiments/adaptive_seacache_overhead_train5_50step_45f_480p/`.
- Overhead experiment design:
  - 5 train prompts: `openvidhd_part1_085`, `openvidhd_part1_086`, `openvidhd_part1_059`, `openvidhd_part1_057`, `openvidhd_part1_016`
  - target PSNRs: `20 25 30`
  - online adaptive run records predictor timing
  - replay run uses the online trace's `(model_stage, branch, step_index) -> threshold` sequence without invoking predictor
  - summary records predictor total elapsed, online/replay compute elapsed, replay-overhead delta, and decision mismatch count
- CPU validation passed for the 5 selected prompts and reusable baselines.
- Launched tmux session `adaptive_seacache_overhead_train5` at 2026-06-19 14:36 CST.
- Overhead session is intentionally waiting for `adaptive_seacache_train15_test5` to finish before loading WanT2V and running.
- Overhead experiment root: `/hy-tmp/wan22_adaptive_seacache_overhead_train5_50step_45f_480p_20260619_143632`

## 2026-06-22 Adaptive SeaCache Result Check

- Added workspace symlinks:
  - `experiment_results/wan22_adaptive_seacache_train10_50step_45f_480p_20260619_134522`
  - `experiment_results/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521`
  - `experiment_results/wan22_adaptive_seacache_overhead_train5_50step_45f_480p_20260619_143632`
- No tmux sessions were running at check time.
- All three experiment roots lack final `results/summary.csv` because the runs did not reach normal completion:
  - `train10`: interrupted pilot, 1/30 candidate completed.
  - `train15_test5`: stopped by OOM at `openvidhd_part1_016 target=20`; 12/60 candidates completed.
  - `overhead_train5`: stopped by OOM at `openvidhd_part1_057 target=30`; 22 rows completed, representing 11/15 online/replay pairs.
- Generated partial summaries from existing artifacts:
  - `/hy-tmp/wan22_adaptive_seacache_train10_50step_45f_480p_20260619_134522/results/partial_summary.csv`
  - `/hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521/results/partial_summary.csv`
  - `/hy-tmp/wan22_adaptive_seacache_overhead_train5_50step_45f_480p_20260619_143632/results/partial_summary.csv`
- Completed video ffprobe checks all passed for existing outputs: `832x480`, `45` frames.
- Key partial results:
  - `train15_test5` over 4 completed train prompts: target 20 mean PSNR `23.544`, speedup `3.226x`; target 25 mean PSNR `24.928`, speedup `2.575x`; target 30 mean PSNR `29.740`, speedup `2.066x`.
  - overhead online/replay completed pairs had zero decision mismatches.
  - predictor timing overhead mean total per candidate: about `0.195s`, about `0.096%` of online compute elapsed.
  - replay-based overhead mean: about `0.218s`, about `0.12%` of online compute elapsed.
- Model weights are not included in this environment artifact; remote machines should fetch or mount Wan2.2 T2V-A14B weights separately at `/hy-tmp/models/Wan2.2-T2V-A14B` or pass the checkpoint path explicitly.
- Current code was pushed to GitHub fork remote `x10ngyx`, branch `main`, at commit `6f68c87`.

## 2026-06-15 TaylorSeer Timestep-Only Prototype

- Added a lightweight timestep-output TaylorSeer prototype for WanT2V experiments.
- New CLI:
  - `--timestep_cache taylorseer`
  - `--taylorseer_interval <int>`
  - `--taylorseer_order <int>`
  - `--taylorseer_ret_steps <int>`
  - `--taylorseer_cutoff_steps <int>`
- Implementation files:
  - `wan/timestep_cache.py`: `TaylorSeerTimestepCacheConfig`, `TaylorSeerTimestepCacheState`, `TaylorSeerTimestepCache`.
  - `wan/text2video.py`: routes timestep-cache branch calls through TaylorSeer when selected, preserving explicit `(model_stage, branch)` keys.
  - `generate.py`: exposes the CLI and constructs the TaylorSeer timestep-cache config.
- Scope note: this is a timestep-output-level Taylor-style forecasting baseline, not a full official TaylorSeer hidden-state/block-level reproduction. It is intended for a first SeaCache-vs-TaylorSeer timestep-only comparison on the existing single-A100 Wan2.2 pipeline.
- Validation:
  - `python -m py_compile generate.py wan/timestep_cache.py wan/text2video.py` passed in the `Wan2.2` conda env.
  - File-level cache behavior check passed without importing the full `wan` package: for `interval=3, order=1, ret_steps=1, cutoff_steps=1`, recompute steps were `[0, 1, 4, 7]` and reuse steps were `[2, 3, 5, 6]`.
- Current instance has no visible GPU: `nvidia-smi` returned `No devices were found`, so real Wan2.2 generation was not launched in this session.

Superseded update:

- The main-code TaylorSeer prototype described above was reverted from `generate.py`, `wan/text2video.py`, `wan/timestep_cache.py`, and `wan/modules/model.py` at the user's request.
- Main Wan2.2 cache paths now contain no TaylorSeer integration and should behave as before for ZEUS, SeaCache, block cache, and CFG cache.
- TaylorSeer work was moved into a standalone implementation under `taylorseer_wan22/`:
  - `taylorseer_wan22/cache.py`
  - `taylorseer_wan22/patch.py`
  - `taylorseer_wan22/text2video.py`
  - `taylorseer_wan22/generate_t2v.py`
- The standalone implementation follows the official TaylorSeer-Wan2.1 structure more closely than the reverted prototype:
  - patch transformer block forward only inside the standalone runner,
  - cache self-attention, cross-attention, and FFN module outputs,
  - use `fresh_threshold`, `max_order`, `first_enhance`, `cache_counter`, and `activated_steps`,
  - keep cond/uncond stream caches separate,
  - keep Wan2.2 high/low model stages separate with independent cache states.
- Validation after the revert and standalone move:
  - `/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m py_compile taylorseer_wan22/cache.py taylorseer_wan22/patch.py taylorseer_wan22/text2video.py taylorseer_wan22/generate_t2v.py generate.py wan/text2video.py wan/timestep_cache.py wan/modules/model.py` passed.
  - `rg -n "TaylorSeer|taylorseer" generate.py wan/text2video.py wan/timestep_cache.py wan/modules/model.py` returned no matches.

## 2026-06-16 Report: SeaCache vs ZEUS-threshold on Ali Prompt 1/2

- Rewrote `reports/report_seacache_vs_zeus_threshold_prompt12.md` as a reader-facing Chinese report.
- The report covers experiment purpose, shared Wan2.2 T2V-A14B configuration, method CLI/threshold settings, data archive roots, prompt 1/2 result tables, and a short conclusion.
- No new inference or PSNR jobs were launched; all numbers were taken from existing archived result tables and prior report data.

## 2026-06-16 Report: Sea CFG Cache vs Original CFG Cache on Ali Prompt 1

- Added `reports/report_cfg_cache_sea_vs_old_prompt01.md` as the second reader-facing Chinese experiment report.

## 2026-06-16 TaylorSeer Third-Party Move And Multi-GPU Prep

- Moved the standalone TaylorSeer Wan2.2 integration from top-level `taylorseer_wan22/` to `third_party/taylorseer_wan22/`.
- Added `third_party/__init__.py` so the runner can be launched as a module:
  - `/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m third_party.taylorseer_wan22.generate_t2v --help`
- Added `third_party/taylorseer_wan22/README.md` with the official-logic alignment notes, Wan2.2 high/low-stage adaptation, and single/multi-GPU launch examples.
- Updated the standalone runner for future multi-GPU use:
  - supports `torchrun`;
  - supports `--ulysses_size <world_size>`;
  - supports `--dit_fsdp` and `--t5_fsdp`;
  - multi-GPU default follows main Wan2.2 behavior by using `offload_model=False` when not explicitly set;
  - rank 0 only saves the output video.
- Updated TaylorSeer patching to handle FSDP-wrapped models by patching the underlying module.
- Current status of the two requested confirmations:
  - Official logic: the third-party implementation follows the public TaylorSeer-Wan2.1 block-level logic: cond stream decides full/Taylor step type, uncond follows it, stream caches are separate, and each full step caches self-attention/cross-attention/FFN module outputs for Taylor prediction. Wan2.2 high/low denoisers are kept as separate cache states; this is a required adaptation because Wan2.2 T2V-A14B uses two denoisers.
  - Multi-GPU: the code path supports torchrun + Ulysses sequence parallel + FSDP for future multi-GPU experiments. This machine currently has no visible GPU, so multi-GPU runtime validation is still pending on the target multi-GPU machine.
- Validation completed on this machine:
  - `/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m py_compile third_party/__init__.py third_party/taylorseer_wan22/__init__.py third_party/taylorseer_wan22/cache.py third_party/taylorseer_wan22/patch.py third_party/taylorseer_wan22/text2video.py third_party/taylorseer_wan22/generate_t2v.py`
  - `/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m third_party.taylorseer_wan22.generate_t2v --help`
  - `rg -n "TaylorSeer|taylorseer" generate.py wan/text2video.py wan/timestep_cache.py wan/modules/model.py` returned no matches; main Wan2.2 cache code remains isolated from TaylorSeer.

## 2026-06-16 TaylorSeer VBench Batch Experiment Scripts

- Added experiment runner directory:
  - `experiments/taylorseer_vbench_50step_45f_480p/`
- Added files:
  - `experiments/__init__.py`
  - `experiments/taylorseer_vbench_50step_45f_480p/__init__.py`
  - `experiments/taylorseer_vbench_50step_45f_480p/run_batch.py`
  - `experiments/taylorseer_vbench_50step_45f_480p/run_tmux.sh`
  - `experiments/taylorseer_vbench_50step_45f_480p/README.md`
- Runner purpose:
  - Run standalone `third_party.taylorseer_wan22` on `test_sets/vbench_every20/prompts.jsonl`.
  - Use project defaults: `t2v-A14B`, `/hy-tmp/models/Wan2.2-T2V-A14B`, seed `42`, `832*480`, `45` frames, `50` DPM++ steps.
  - Load the Wan2.2/TaylorSeer pipeline once per process and run selected VBench prompts sequentially.
  - Reset TaylorSeer cache at the start of every generated sample so prompts do not share module-output cache state.
  - Support multi-GPU `torchrun` with `--ulysses_size`, `--dit_fsdp`, and `--t5_fsdp`.
- Archive behavior:
  - Writes videos, per-sample logs, command records, ffprobe JSON, `results/summary.csv`, `results/summary.json`, failed records, `experiment_config.json`, `launch.env`, `gpu.txt`, and `runner.log`.
  - This TaylorSeer-only VBench runner does not generate no-cache baselines; PSNR and speedup fields are intentionally blank with an explanatory note. Generate a separate no-cache baseline run if PSNR/speedup is required.
- Validation completed on this machine:
  - `/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m py_compile experiments/__init__.py experiments/taylorseer_vbench_50step_45f_480p/__init__.py experiments/taylorseer_vbench_50step_45f_480p/run_batch.py third_party/taylorseer_wan22/cache.py third_party/taylorseer_wan22/text2video.py third_party/taylorseer_wan22/generate_t2v.py`
  - `/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m experiments.taylorseer_vbench_50step_45f_480p.run_batch --cpu_validate --prompt_limit 2`
  - `/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m experiments.taylorseer_vbench_50step_45f_480p.run_batch --help`
- The report compares CFG-only `--cfg_cache threshold` against SeaCache-style `--cfg_cache sea-threshold` on Ali prompt 1.
- Included shared generation settings, CFG cache parameter settings, archive roots, result tables with speedup/PSNR/reuse counts, per-stage CFG reuse counts, and a short conclusion.
- No new inference or PSNR jobs were launched; all numbers were taken from `/hy-tmp/wan22_cfg_cache_prompt01_50step_45f_480p_20260613_163243`.

## 2026-06-16 Report: Sea Block Cache vs Original Block-Group Cache on Ali Prompt 1

- Added `reports/report_block_cache_sea_vs_old_prompt01.md` as the third reader-facing Chinese experiment report.
- The report compares block-cache-only original `block-group` with `pooled_rel_l1` against SeaCache-style `block-group` with `sea_full_rel_l1` and `accumulated` decision mode on Ali prompt 1.
- Included shared generation settings, block cache parameter settings, archive roots, result tables with speedup/PSNR/reuse counts, Sea block high/low stage reuse counts, and a short conclusion.
- Used original block-group results from `/hy-tmp/wan22_block_cache_only_50step_45f_480p_20260609_125436` and Sea block results from `/hy-tmp/wan22_block_group_sea_full_prompt01_50step_45f_480p_20260614_235605`.
- Noted that `/hy-tmp/wan22_block_group_sea_full_prompt01_50step_45f_480p_20260613_235449` was an early failed pilot and is not used as a result source.
- No new inference or PSNR jobs were launched.

## 2026-06-16 Report: Sea-Style Three-Cache Threshold Grid on Ali Prompt 1

- Added `reports/report_three_cache_sea_threshold_grid_prompt01.md` as the fourth reader-facing Chinese experiment report.
- The report summarizes the completed sea-style three-cache grid on Ali prompt 1:
  - timestep cache: `seacache`
  - block cache: `block-group` with `sea_full_rel_l1` and `accumulated`
  - CFG cache: `sea-threshold`
- Included shared generation settings, cache order, per-cache threshold/config settings, archive roots, completion status, best candidates by PSNR target, representative combinations, single-threshold-dimension trends, PSNR distribution, and a short conclusion.
- Used result root `/hy-tmp/wan22_three_cache_sea_prompt01_50step_45f_480p_20260614_005404`.
- No new inference or PSNR jobs were launched; summary statistics were computed from the existing `results/summary.csv`.
- Added the complete 125-row result table to the report with candidate label, three thresholds, elapsed time, speedup, PSNR/min PSNR, and timestep/block/CFG reuse/recompute counts.
- Updated all four reader-facing reports so their configuration tables use English prompt summaries, matching the original Ali prompt text instead of Chinese paraphrases.

## 2026-06-16 Report: Four-Experiment Summary

- Added `reports/report_cache_experiments_summary.md` as a combined reader-facing report for the four cache experiments:
  - SeaCache vs ZEUS-threshold on Ali Prompt 1/2.
  - Sea CFG cache vs original CFG cache on Ali Prompt 1.
  - Sea block cache vs original block-group cache on Ali Prompt 1.
  - Sea-style timestep/block/CFG three-cache threshold grid on Ali Prompt 1.
- Kept experiment configurations and the key complete result tables in the report.
- Kept the full 125-row three-cache grid result table as an appendix, while removing secondary trend/distribution statistics from the summary body.
- No new inference, PSNR, or GPU jobs were launched.

## 2026-06-15 Adaptive Threshold Predictor Scaffold

- Created `adaptive_threshold_predictor/` as the isolated workspace for prediction-network code.
- Inspected `/hy-tmp/openvid_100_seacache_trace_data/data`; `manifest.json` reports 100 samples and 1000 SeaCache candidates.
- Confirmed traced baseline step latent shape is `[16, 12, 60, 104]` in single-step `.pt` files, with `meta.pt` storing 50 timesteps.
- Added timestep-cache-only `ImprovedAdaCacheGate`:
  - Inputs: latent, timestep, target PSNR.
  - Static branch: `AdaptiveAvgPool3d((2, 2, 2))` on latent.
  - Dynamic branch: first-order absolute temporal difference plus the same pool.
  - Condition branch: lightweight MLP over timestep and target PSNR.
  - Output: one Sigmoid threshold in `[0, 1]`.
  - Current default model has about 29K trainable parameters for 16 latent channels and hidden dim 64.
- Added trace data utilities and a direct-threshold label builder for initial supervised training: for each sample and target PSNR, use the fastest threshold whose measured PSNR reaches the target, or the highest-PSNR threshold if unreachable.
- Verified:
  - `python -m adaptive_threshold_predictor.inspect_trace_data`
  - `python -m adaptive_threshold_predictor.train_gate --epochs 1 --batch_size 2 --max_examples 8 --out_dir /hy-tmp/wan22_adaptive_threshold_predictor_smoke`
- Smoke output saved under `/hy-tmp/wan22_adaptive_threshold_predictor_smoke`; no generation runner or cache core logic was changed.

## 2026-06-15 Adaptive Feature Ablation Interface

- Updated `ImprovedAdaCacheGate` so timestep and target PSNR are always input through the same condition branch, while exactly one latent-derived feature set is selected for comparison.
- Kept the prediction head and latent feature output dimension fixed across feature sets, so validation-loss differences reflect feature information rather than architecture size.
- Supported feature sets:
  - `latent_pool`
  - `temporal_mean`
  - `temporal_var`
  - `frame_diff_mean`
  - `frame_diff_var`
- Added `--feature_set` to `adaptive_threshold_predictor.train_gate`.
- Added `adaptive_threshold_predictor.run_feature_ablation` to run all feature sets and write `feature_ablation_summary.json`.
- Smoke validation:
  - `/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m adaptive_threshold_predictor.run_feature_ablation --epochs 1 --batch_size 2 --max_examples 10 --device cpu --out_root /hy-tmp/wan22_adaptive_threshold_feature_ablation_smoke`
  - All five feature sets had the same trainable parameter count (`21057`) and completed forward/backward/save.
  - The smoke run is only a functionality check, not a quantitative conclusion because it used 10 examples and 1 epoch.

## 2026-06-16 Adaptive Predictor Split And Conditioning Update

- Updated adaptive predictor training to split train/validation by `sample_id`; all step/target-PSNR examples from the same sample now stay on the same side of the split.
- Changed timestep conditioning to use 50-step denoising progress rather than raw scheduler timestep:
  - dataset now passes `step_index / (num_steps - 1)`, so the input is already in `[0, 1]`.
  - model now clamps timestep input to `[0, 1]` instead of dividing by `1000`.
- Kept the current sample-level oracle label construction unchanged by request.
- Did not add high/low stage conditioning and did not add optical-flow features.
- Updated `ImprovedAdaCacheGate` with a fixed `feature_proj`:
  - selected latent-derived feature is pooled and projected to `hidden_dim`;
  - condition embedding remains `hidden_dim`;
  - prediction head always receives `2 * hidden_dim`, keeping the downstream architecture fixed across feature sets and grid-size experiments.
- Validation:
  - `python -m py_compile adaptive_threshold_predictor/models.py adaptive_threshold_predictor/data.py adaptive_threshold_predictor/train_gate.py adaptive_threshold_predictor/inspect_trace_data.py adaptive_threshold_predictor/run_feature_ablation.py`
  - group split smoke check with `max_examples=120`: train/val sample overlap was `0`, first timestep fraction was `0.0`.
  - feature-ablation smoke run completed on CPU with `max_examples=60`; all five feature sets had identical parameter count (`29377`).
  - Smoke output: `/hy-tmp/wan22_adaptive_threshold_feature_ablation_smoke_v2/feature_ablation_summary.json`.

## 2026-06-16 Adaptive Predictor Full-Step Default

- Changed direct-threshold dataset defaults to use all 50 denoising steps instead of 7 sampled steps.
- Current direct-threshold example count is `100 samples * 6 target_psnr values * 50 steps = 30000`.
- Clarified that `100 * 10 * 50 = 50000` corresponds to a different candidate/metric-prediction formulation where each measured threshold candidate is also an input row; the current direct-threshold formulation selects one oracle threshold per `sample_id + target_psnr`.
- Exposed PSNR normalization bounds in training scripts:
  - `--psnr_min`, default `10.0`
  - `--psnr_max`, default `50.0`
- Current condition normalization:
  - `timestep = step_index / 49`
  - `target_psnr_norm = clamp((target_psnr - psnr_min) / (psnr_max - psnr_min), 0, 1)`
- Verified default dataset construction:
  - examples: `30000`
  - train/val examples: `24000/6000`
  - train/val samples: `80/20`
  - train/val sample overlap: `0`
- Smoke training passed:
  - `/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m adaptive_threshold_predictor.train_gate --feature_set latent_pool --epochs 1 --batch_size 2 --max_examples 20 --device cpu --out_dir /hy-tmp/wan22_adaptive_threshold_train_smoke_v3`

## 2026-06-16 Candidate-Inverse Dataset Mode

- Added `candidate_inverse` dataset mode and made it the default for adaptive threshold predictor training.
- `candidate_inverse` rows are built from each measured SeaCache threshold candidate and each denoising step:
  - input latent: `data/seacache/step_inputs/<threshold_label>/<sample_id>/step_*.pt`
  - input timestep: `step_index / 49`
  - input target PSNR: the candidate run's achieved `mean_psnr`
  - label: the threshold used by that candidate run
- Kept `target_oracle` mode available for comparison.
- Verified default dataset construction:
  - mode: `candidate_inverse`
  - examples: `50000`
  - train/val examples: `40000/10000`
  - train/val samples: `80/20`
  - train/val sample overlap: `0`
  - each of the 10 thresholds contributes `5000` examples.
- Smoke training passed:
  - `/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m adaptive_threshold_predictor.train_gate --epochs 1 --batch_size 2 --max_examples 30 --device cpu --out_dir /hy-tmp/wan22_adaptive_threshold_candidate_inverse_smoke`

## 2026-06-16 Adaptive Feature Cache And First Ablation

- Raw latent training was too slow because it repeatedly opened 50,000 step `.pt` files.
- Added cached feature support:
  - `adaptive_threshold_predictor/build_feature_cache.py`
  - `CachedFeatureThresholdDataset`
  - `CachedFeatureAdaCacheGate`
  - `--cache_dir` support in `train_gate.py` and `run_feature_ablation.py`.
- Built full candidate-inverse feature cache:
  - cache root: `/hy-tmp/wan22_adaptive_threshold_feature_cache_candidate_inverse_20260616_012409`
  - examples: `50000`
  - feature sets: `latent_pool`, `temporal_mean`, `temporal_var`, `frame_diff_mean`, `frame_diff_var`
  - each feature tensor shape: `[50000, 128]`
  - total cache size: about `124M`
  - elapsed: `300.38s`
- Ran cached five-feature ablation:
  - result root: `/hy-tmp/wan22_adaptive_threshold_feature_ablation_cached_20260616_012409`
  - dataset mode: `candidate_inverse`
  - split: group by `sample_id`, `80/20` samples
  - epochs: `3`
  - batch size: `256`
  - hidden dim: `64`
  - all models: `29377` trainable parameters
  - saved per-feature `config.json`, `split.json`, `best_model.pt`, `final_model.pt`, `metrics.json`, and `val_predictions.csv`.
- Best validation-loss ranking:
  - `temporal_mean`: best epoch `2`, val loss `0.012259`, val MAE `0.120107`
  - `latent_pool`: best epoch `2`, val loss `0.012755`, val MAE `0.116558`
  - `frame_diff_mean`: best epoch `3`, val loss `0.014569`, val MAE `0.132957`
  - `temporal_var`: best epoch `1`, val loss `0.014595`, val MAE `0.129695`
  - `frame_diff_var`: best epoch `2`, val loss `0.014659`, val MAE `0.131198`
- Summary files:
  - `/hy-tmp/wan22_adaptive_threshold_feature_ablation_cached_20260616_012409/feature_ablation_summary.csv`
  - `/hy-tmp/wan22_adaptive_threshold_feature_ablation_cached_20260616_012409/feature_ablation_summary.json`
  - `/hy-tmp/wan22_adaptive_threshold_feature_ablation_cached_20260616_012409/feature_ablation_best_summary.csv`
  - `/hy-tmp/wan22_adaptive_threshold_feature_ablation_cached_20260616_012409/feature_ablation_best_summary.json`

## 2026-06-13 Documentation Cleanup

- Simplified `AGENTS.md` by merging the former `代码与接口约定` and `数据与自适应阶段规划` sections into `项目目标`.
- Kept only the necessary cache composition, unified CLI, dataset-row, and adaptive predictor constraints in the project-goal section; no code or experiment logic was changed.

## 2026-06-13 OpenVid SeaCache Inspection

- Inspected `/hy-tmp/wan22_seacache_openvid100_50step_45f_480p_20260612_002814`.
- Confirmed the run completed cleanly with no files under `failed/`.
- Artifact counts: `25` baseline MP4s, `25` MP4s for each of `10` SeaCache thresholds, `275` ffprobe JSON files, `750` PSNR files, `551` run logs, and `275` command files.
- `results/summary.csv` contains `250` candidate rows across `25` unique samples and `10` threshold labels.
- `results/aggregate_by_threshold.csv` contains the completed speed/quality frontier for this OpenVid shard.

## 2026-06-13 Sea-Style CFG Cache Implementation

- Added `SeaCFGCacheConfig` and `SeaCFGCache` in `wan/cfg_cache.py`.
- Added CLI method `--cfg_cache sea-threshold` in `generate.py`.
- Wired `wan/text2video.py` so the new CFG cache mode uses `model.seacache_feature(...)`, scheduler sigmas, accumulated SEA-filtered relative-L1, and first/tail-step recompute protection.
- Existing `--cfg_cache threshold` behavior is unchanged.
- Validation run:
  - `/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m py_compile wan/cfg_cache.py wan/text2video.py generate.py`
  - `/hy-tmp/miniconda3/envs/Wan2.2/bin/python generate.py --help | rg -n "sea-threshold|cfg_sea|cfg_ret|cfg_cutoff"`
  - A small CPU state-machine check confirmed first-step protection, tail-step protection, accumulated distance tracking, and consecutive reuse without `cfg_max_reuse` limiting `sea-threshold`.

## 2026-06-13 CFG-Only Prompt-01 Comparison Launch

- Added prompt-01 cfg-cache-only experiment runner under `experiments/cfg_cache_prompt01_50step_45f_480p/`.
- Launched tmux session `cfg_cache_p01_20260613_163243`; it completed cleanly and tmux exited.
- Result root: `/hy-tmp/wan22_cfg_cache_prompt01_50step_45f_480p_20260613_163243`.
- Workspace symlink: `experiment_results/wan22_cfg_cache_prompt01_50step_45f_480p_20260613_163243`.
- Baseline is reused from `/hy-tmp/wan22_cache_ablation_prompt01_50step_45f_480p_20260609_184625/baseline/prompt_01.mp4`; baseline compute seconds are `522.603`.
- Candidate matrix:
  - old CFG `threshold:0.02`
  - old CFG `threshold:0.03`
  - new CFG `sea-threshold:0.10`
  - new CFG `sea-threshold:0.20`
  - new CFG `sea-threshold:0.30`
- Cache settings: `--timestep_cache none`, `--block_cache none`, CFG only.
- Runner validated before launch:
  - `/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m py_compile experiments/cfg_cache_prompt01_50step_45f_480p/run_batch.py wan/cfg_cache.py wan/text2video.py generate.py`
  - `/hy-tmp/miniconda3/envs/Wan2.2/bin/python experiments/cfg_cache_prompt01_50step_45f_480p/run_batch.py --cpu_validate`
  - `bash -n experiments/cfg_cache_prompt01_50step_45f_480p/run_tmux.sh`
- Launch check: tmux was active, GPU was loaded, and the first candidate `threshold_th_0p02` had entered 50-step sampling.
- Completion check: no failed files; all `5/5` candidates have videos, ffprobe JSON, PSNR JSON, logs, and command records.
- Result tables:
  - `/hy-tmp/wan22_cfg_cache_prompt01_50step_45f_480p_20260613_163243/results/summary.csv`
  - `/hy-tmp/wan22_cfg_cache_prompt01_50step_45f_480p_20260613_163243/results/summary_with_cache.csv`
- Results:
  - old `threshold 0.02`: `1.041x`, mean PSNR `26.732 dB`, min PSNR `22.89 dB`, CFG reuse/recompute `9/41`.
  - old `threshold 0.03`: `1.137x`, mean PSNR `21.571 dB`, min PSNR `20.31 dB`, CFG reuse/recompute `17/33`.
  - new `sea-threshold 0.10`: `1.007x`, mean PSNR `37.457 dB`, min PSNR `34.81 dB`, CFG reuse/recompute `6/44`.
  - new `sea-threshold 0.20`: `1.175x`, mean PSNR `26.226 dB`, min PSNR `23.13 dB`, CFG reuse/recompute `20/30`.
  - new `sea-threshold 0.30`: `1.297x`, mean PSNR `21.359 dB`, min PSNR `20.07 dB`, CFG reuse/recompute `28/22`.
- Takeaway on prompt-01: Sea CFG `0.10` is a very high-quality conservative point; Sea CFG `0.20` improves speed over old CFG `0.03` while preserving quality close to old CFG `0.02`; Sea CFG `0.30` is the aggressive point with similar quality to old `0.03` but higher speed.

## 2026-06-13 Sea Timestep + Sea CFG No-Skip-Accum Prompt-01

- Added prompt-01 sea timestep + sea CFG experiment runner under `experiments/timestep_cfg_prompt01_50step_45f_480p/`.
- Chosen composition behavior: when CFG cache reuses and skips `uncond`, the skipped `uncond` branch does not advance sea timestep accumulated distance.
- Candidate grid is `2x2`: sea timestep thresholds `0.10 0.20` crossed with sea CFG thresholds `0.10 0.20`.
- Baseline is reused from `/hy-tmp/wan22_cache_ablation_prompt01_50step_45f_480p_20260609_184625`.
- Validation before relaunch:
  - `/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m py_compile wan/timestep_cache.py wan/text2video.py wan/cfg_cache.py generate.py experiments/timestep_cfg_prompt01_50step_45f_480p/run_batch.py`
  - `/hy-tmp/miniconda3/envs/Wan2.2/bin/python experiments/timestep_cfg_prompt01_50step_45f_480p/run_batch.py --cpu_validate`
- tmux session: `timestep_cfg_noaccum_p01_20260613_213000`.
- Result root: `/hy-tmp/wan22_timestep_cfg_prompt01_no_uncond_skip_accum_50step_45f_480p_20260613_213000`.
- Workspace symlink: `experiment_results/wan22_timestep_cfg_prompt01_no_uncond_skip_accum_50step_45f_480p_20260613_213000`.
- Launch check: process was running, baseline artifacts copied, and `pipeline_init.log` had started writing.
- Completion check on 2026-06-14: runner ended with `Completed experiment`; `failed/` is empty; all `4/4` candidates have videos, ffprobe JSON, PSNR artifacts, logs, command records, and result rows.
- All baseline and candidate videos were verified as `832x480`, `45` frames, `16 fps`, duration `2.8125s`.
- Results:
  - timestep `0.10` + CFG `0.10`: `1.067x`, mean PSNR `36.747 dB`, min PSNR `35.04 dB`.
  - timestep `0.10` + CFG `0.20`: `1.256x`, mean PSNR `26.430 dB`, min PSNR `23.14 dB`.
  - timestep `0.20` + CFG `0.10`: `1.498x`, mean PSNR `24.433 dB`, min PSNR `21.84 dB`.
  - timestep `0.20` + CFG `0.20`: `1.550x`, mean PSNR `24.848 dB`, min PSNR `22.11 dB`.
- Current prompt-01 takeaway: the selected No-Skip-Accum behavior is the maintained sea timestep + sea CFG composition. It does not clearly beat sea timestep-only `0.20` (`1.569x`, `24.558 dB`) on speed, but `0.20+0.20` is close (`1.550x`, `24.848 dB`) and slightly improves PSNR.

## 2026-06-14 Skip-Accounting Cleanup

- User selected No-Skip-Accum as the final behavior for sea timestep + sea CFG.
- Removed the alternative skip-accounting implementation from code:
  - deleted `advance_skipped_branch(...)` and related SeaCache timestep state fields from `wan/timestep_cache.py`.
  - deleted the unused CFG-skip helper from `wan/text2video.py`.
  - removed skip-accounting columns from `experiments/timestep_cfg_prompt01_50step_45f_480p/run_batch.py`.
- Deleted the skip-accounting restore point and superseded prompt-01 skip-accounting experiment archive/symlink.

## 2026-06-13 Block-Group Sea Full Accumulated Cache Implementation

- Added an experimental full-feature Sea-style block-group cache path.
- New CLI/config fields:
  - `--block_group_decision {instant,accumulated}`; default `instant` preserves legacy behavior.
  - `--block_group_metric sea_full_rel_l1`; existing `pooled_rel_l1` and `full_rel_l1` remain available.
  - `--block_group_ret_steps`, `--block_group_cutoff_steps`, `--block_group_sea_power_exp`, `--block_group_sea_power_const`, `--block_group_sea_norm_mode`.
- `sea_full_rel_l1` computes each block group indicator from the full group-entry modulated norm feature, reshapes it to the latent token grid, applies the same scheduler-aware SEA frequency filter style as SeaCache, and stores the filtered full feature for later distance checks.
- `accumulated` decision mode accumulates per-step relative-L1 distance and reuses the group residual while the accumulated distance remains below `--block_threshold`; first/tail step recompute protection is controlled by the new ret/cutoff args.
- `wan/modules/model.py` now passes `grid_sizes[0]` and scheduler sigmas into block-group cache decisions.
- Important risk: this mode intentionally stores full filtered features per block group in addition to cached residuals, so it can add multiple GB of GPU memory pressure at the default 832x480/45f shape. Use first with small thresholds or small pilot runs before large grids.
- Validation run:
  - `/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m py_compile wan/block_group_cache.py wan/modules/model.py wan/text2video.py generate.py`
  - `/hy-tmp/miniconda3/envs/Wan2.2/bin/python generate.py --help | rg -n "block_group_(decision|metric|ret|cutoff|sea)"`
  - CPU state-machine check confirmed first-step protection, reuse under accumulated threshold, tail-step recompute, pending filtered feature storage, and summary fields.

## 2026-06-14 Three Sea-Style Cache Prompt-01 Grid Launch

- Added prompt-01 three-cache sea-style grid runner under `experiments/three_cache_sea_prompt01_50step_45f_480p/`.
- Cache methods:
  - timestep cache: `seacache`
  - block cache: `block-group` with `metric=sea_full_rel_l1`, `decision=accumulated`, `group_size=5`, `max_reuse=50`, `ret_steps=1`, `cutoff_steps=1`
  - CFG cache: `sea-threshold`, `ret_steps=1`, `cutoff_steps=1`
- Cache order remains the project-standard composition: CFG cache outermost; actual cond/uncond branch first checks timestep cache; block-group cache only runs on timestep-cache miss.
- Threshold grid: each cache uses `0.05 0.10 0.20 0.40 1.00`, for `5*5*5 = 125` prompt-01 candidates. This spans near-no-reuse through very aggressive reuse for the current sea-style metrics.
- Baseline is reused from `/hy-tmp/wan22_cache_ablation_prompt01_50step_45f_480p_20260609_184625`; baseline compute seconds are `522.603`.
- Validation before launch:
  - `/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m py_compile experiments/three_cache_sea_prompt01_50step_45f_480p/run_batch.py wan/timestep_cache.py wan/block_group_cache.py wan/cfg_cache.py wan/text2video.py generate.py`
  - `/hy-tmp/miniconda3/envs/Wan2.2/bin/python experiments/three_cache_sea_prompt01_50step_45f_480p/run_batch.py --cpu_validate`
  - `bash -n experiments/three_cache_sea_prompt01_50step_45f_480p/run_tmux.sh`
- Launched tmux session: `three_cache_sea_p01_20260614_005404`.
- Result root: `/hy-tmp/wan22_three_cache_sea_prompt01_50step_45f_480p_20260614_005404`.
- Workspace symlink: `experiment_results/wan22_three_cache_sea_prompt01_50step_45f_480p_20260614_005404`.
- Launch check: tmux is active; first candidate `sea_ts_0p05__sea_bg_0p05__sea_cfg_0p05` entered 50-step sampling; GPU showed about `63107 MiB` used and `100%` utilization.
- Runner writes per-candidate videos, logs, command records, ffprobe JSON, PSNR JSON/logs, and continuously refreshes `results/summary.csv` / `results/summary.json` after each completed candidate.
- Known risk: block sea-full cache stores full filtered group features, so high-threshold combinations may hit GPU memory pressure. If the run fails, inspect `failed/` and the last candidate log, then resume with `RESUME_EXISTING=True`.

## 2026-06-15 Three Sea-Style Cache Prompt-01 Grid Completion

- Checked `/hy-tmp/wan22_three_cache_sea_prompt01_50step_45f_480p_20260614_005404`.
- tmux has exited; GPU is idle; `failed/` is empty; `runner.log` ends with `Completed experiment`.
- Completed artifacts:
  - `125/125` videos
  - `125/125` candidate ffprobe JSON files
  - `125/125` candidate PSNR JSON files
  - `125/125` candidate logs
  - `125/125` command records
  - result tables: `results/summary.csv` and `results/summary.json`
- All candidate ffprobe rows match `832x480`, `45` frames, `16 fps`, duration `2.812500s`.
- PSNR rows: `124` finite rows and `1` all-perfect/Infinity row (`sea_ts_0p05__sea_bg_0p05__sea_cfg_0p05`).
- Fastest finite candidate: `sea_ts_1p00__sea_bg_1p00__sea_cfg_1p00`, `5.644x`, PSNR `11.914 dB`.
- Best finite PSNR candidate: `sea_ts_0p05__sea_bg_0p10__sea_cfg_0p05`, `0.987x`, PSNR `37.465 dB`.
- Best speed by PSNR target:
  - PSNR `>=35 dB`: `sea_ts_0p10__sea_bg_0p10__sea_cfg_0p10`, `1.025x`, PSNR `36.747 dB`
  - PSNR `>=30 dB`: `sea_ts_0p10__sea_bg_0p10__sea_cfg_0p10`, `1.025x`, PSNR `36.747 dB`
  - PSNR `>=26 dB`: `sea_ts_0p10__sea_bg_0p05__sea_cfg_0p20`, `1.208x`, PSNR `26.430 dB`
  - PSNR `>=24 dB`: `sea_ts_0p20__sea_bg_0p20__sea_cfg_0p20`, `1.496x`, PSNR `24.898 dB`
  - PSNR `>=20 dB`: `sea_ts_0p20__sea_bg_0p20__sea_cfg_0p20`, `1.496x`, PSNR `24.898 dB`
  - PSNR `>=19 dB`: `sea_ts_0p40__sea_bg_0p10__sea_cfg_1p00`, `2.845x`, PSNR `19.007 dB`
  - PSNR `>=18 dB`: `sea_ts_1p00__sea_bg_0p05__sea_cfg_0p20`, `3.575x`, PSNR `18.233 dB`
  - PSNR `>=16 dB`: `sea_ts_0p40__sea_bg_1p00__sea_cfg_0p40`, `3.895x`, PSNR `16.662 dB`
  - PSNR `>=15 dB`: `sea_ts_1p00__sea_bg_1p00__sea_cfg_0p20`, `4.873x`, PSNR `15.633 dB`
- Main prompt-01 takeaway: the three-cache sea-style grid completed without OOM. The useful higher-quality frontier is still dominated by moderate thresholds around `0.10-0.20`; aggressive thresholds reach much higher speed but quality falls quickly.

## 2026-06-15 OSS OpenVid Space Check

- Checked disk space before considering OSS download:
  - `/hy-tmp`: `400G` total, `265G` used, `136G` available.
  - `/`: `30G` total, `7.7G` used, `23G` available.
- OSS login for the provided HyCloud account succeeded after the previous token returned `401 Authentication Failed`.
- `oss://datasets/` currently contains OpenVid prompt data split as:
  - `prompt001-033.tar.gz`: `40.59GB`
  - `prompt034-100.tar.gz`: `82.61GB`
  - Combined compressed size: about `123.20GB`.
- Conclusion: downloading only the two compressed prompt archives to `/hy-tmp` is technically possible but leaves only about `12-13GB` free, which is too tight for normal work and not enough to safely extract them. Do not download and extract both archives on the current disk without first freeing substantial space or using a streaming/selective extraction workflow.

## 2026-06-15 OpenVid Prompt Archive Download

- User requested downloading both OSS prompt archives despite tight disk margin.
- Added script: `scripts/download_openvid_prompt_archives.sh`.
- Launched tmux session: `download_openvid_prompts`.
- Log path: `logs/2026-06-15_openvid_prompt_archive_download.log`.
- Download order:
  1. `oss://datasets/prompt001-033.tar.gz` -> `/hy-tmp/prompt001-033.tar.gz`
  2. `oss://datasets/prompt034-100.tar.gz` -> `/hy-tmp/prompt034-100.tar.gz`
- Launch state: first archive was downloading; log showed about `3.9%` of `40.59GB`. `/hy-tmp` had about `134G` available at launch after partial temp/download state.
- Important: do not start extraction while both archives are being downloaded unless space is freed first.

## 2026-06-15 OpenVid Prompt Archive Extraction

- `/hy-tmp` was expanded to `600G`; before extraction it had about `212G` available.
- Confirmed downloaded archives:
  - `/hy-tmp/prompt001-033.tar.gz`: `41G`
  - `/hy-tmp/prompt034-100.tar.gz`: `83G`
- Added extraction/organization script: `scripts/extract_openvid_prompt_archives.sh`.
- Launched tmux session: `extract_openvid_prompts`; it completed successfully.
- Extraction log: `logs/2026-06-15_openvid_prompt_archive_extract.log`.
- Unified extracted root: `/hy-tmp/openvid_100_seacache_trace_data`
- Workspace symlink: `experiment_results/openvid_100_seacache_trace_data`
- Organized helper symlinks:
  - `/hy-tmp/openvid_100_seacache_trace_data/sources/`: 2 source experiment directories.
  - `/hy-tmp/openvid_100_seacache_trace_data/shards/`: 6 shard directories covering prompt indices `000-099`.
- Final validation counts:
  - source dirs: `2`
  - shard dirs: `6`
  - baseline videos: `100`
  - SeaCache videos: `1000`
  - per-shard `results/summary.csv`: `6`
- Final extracted directory size: `135G`.
- Final `/hy-tmp` state after keeping both compressed archives and extracted data: `600G` total, `523G` used, `78G` available.

## 2026-06-15 OpenVid Training Data Layout

- User requested a clean `data/` layout for downstream training, without exposing the original `001-033`, `034-100`, or shard splits.
- Added builder script: `scripts/build_openvid_training_data_layout.py`.
- Created flat training-data view under `/hy-tmp/openvid_100_seacache_trace_data/data`.
- The `data/` directory uses stable symlinks rather than duplicating the 135G extracted payload.
- Public training entry points:
  - `/hy-tmp/openvid_100_seacache_trace_data/data/tables/summary.csv`
  - `/hy-tmp/openvid_100_seacache_trace_data/data/tables/summary.jsonl`
  - `/hy-tmp/openvid_100_seacache_trace_data/data/tables/prompts.csv`
  - `/hy-tmp/openvid_100_seacache_trace_data/data/tables/prompts.jsonl`
  - `/hy-tmp/openvid_100_seacache_trace_data/data/metadata/manifest.json`
- Flat artifact layout:
  - `data/baseline/videos/<sample_id>.mp4`
  - `data/baseline/logs/<sample_id>.log`
  - `data/baseline/ffprobe/<sample_id>.json`
  - `data/baseline/commands/<sample_id>.sh`
  - `data/baseline/step_inputs/<sample_id>/`
  - `data/seacache/videos/th_<threshold>/<sample_id>.mp4`
  - `data/seacache/logs/th_<threshold>/<sample_id>.log`
  - `data/seacache/ffprobe/th_<threshold>/<sample_id>.json`
  - `data/seacache/psnr/th_<threshold>/<sample_id>.json`
  - `data/seacache/commands/th_<threshold>/<sample_id>.sh`
  - `data/seacache/step_inputs/th_<threshold>/<sample_id>/`
- Validation:
  - `summary.csv`: `1000` candidate rows.
  - `prompts.csv`: `100` prompt rows.
  - baseline video links: `100`.
  - SeaCache video links: `1000`.
  - PSNR JSON links: `1000`.
  - step input links: `1100`.
  - broken links under `data/`: `0`.
  - required paths in `summary.csv`: `0` missing.
  - public required paths in `summary.csv` contain no `001_033`, `034_100`, or `shard` split names.
  - One optional PSNR text log and ffmpeg log are absent for `openvidhd_part1_033` at `th_0p10`; the PSNR JSON exists and the optional table fields are left empty.

## 2026-06-15 Test Set Prompt Resource Organization

- Created consolidated prompt resource directory: `test_sets/`.
- Organized three prompt sets:
  - `test_sets/ali_10/`: 10 Ali prompts copied from repository `prompt.txt`.
  - `test_sets/openvid_100/`: 100 OpenVid prompts extracted from `/hy-tmp/openvid_100_wan22_prompts.zip`; source metadata files from the zip were preserved.
  - `test_sets/vbench_every20/`: VBench-2.0 prompts downloaded from `https://raw.githubusercontent.com/Vchitect/VBench/master/VBench-2.0/prompts/VBench2_full_text.txt`; sampled source prompt lines `1, 21, 41, ... 1001`, producing 51 prompts from 1013 source prompt lines.
- Each set has both `prompts.txt` for runner input and `prompts.jsonl` with stable `sample_id`, source index, and text.
- Added combined indexes:
  - `test_sets/all_prompts.jsonl`: 161 rows.
  - `test_sets/all_prompts.csv`: 161 data rows plus header.
  - `test_sets/manifest.json`: source paths, counts, files, and VBench sampling rule.
  - `test_sets/SHA256SUMS`: checksums for all prompt-resource files.
- Validation run:
  - `wc -l` confirmed Ali `10`, OpenVid `100`, VBench sampled `51`, combined JSONL `161`.
  - `python -m json.tool test_sets/manifest.json` passed.
  - All JSONL files under `test_sets/` parsed successfully.
- Follow-up AGENTS update: replaced the old OpenVid zip-only resource note with the unified prompt test set directory `/hy-tmp/work/Wan2.2/test_sets`; OpenVid-100 prompt files are now documented as `test_sets/openvid_100/`.

## 2026-06-16 Adaptive Threshold Predictor Pooling Grid Ablation

- Continued timestep-threshold predictor work under `adaptive_threshold_predictor/`.
- Ran larger pooling-grid ablation for cached candidate-inverse training:
  - output root: `/hy-tmp/wan22_adaptive_threshold_grid_ablation_20260616_020314`
  - grids: `2x4x4`, `3x4x4`, `4x4x4`
  - dataset mode: `candidate_inverse`
  - train/val split: grouped by `sample_id`
  - examples: `50000` total, `40000` train, `10000` val
  - epochs: `3`
  - batch size: `256`
  - feature sets: `latent_pool`, `temporal_mean`, `temporal_var`, `frame_diff_mean`, `frame_diff_var`
- Built one feature cache per grid:
  - `2x4x4`: feature dim `512`, model params `53953`
  - `3x4x4`: feature dim `768`, model params `70337`
  - `4x4x4`: feature dim `1024`, model params `86721`
- Summary tables:
  - `/hy-tmp/wan22_adaptive_threshold_grid_ablation_20260616_020314/grid_feature_ablation_best_summary.csv`
  - `/hy-tmp/wan22_adaptive_threshold_grid_ablation_20260616_020314/grid_feature_ablation_best_summary.json`
- Best validation-loss results compared with the existing `2x2x2` cache:
  - `2x2x2 temporal_mean`: best val loss `0.012259`, val MAE `0.120107`, params `29377`
  - `2x4x4 latent_pool`: best val loss `0.012434`, val MAE `0.118093`, params `53953`
  - `4x4x4 latent_pool`: best val loss `0.012733`, val MAE `0.118652`, params `86721`
  - `2x2x2 latent_pool`: best val loss `0.012755`, val MAE `0.116558`, params `29377`
  - `3x4x4 temporal_mean`: best val loss `0.013236`, val MAE `0.124000`, params `70337`
- Current takeaway: increasing the pooling grid did not improve best validation loss in this 3-epoch single-split run. `2x4x4 latent_pool` is close and has slightly lower MAE than the best `2x2x2 temporal_mean`, but the larger grids show more early overfitting and higher parameter count. Keep `2x2x2 temporal_mean` as the current loss-based default unless a longer/multi-seed run changes the ranking.

## 2026-06-16 Adaptive Predictor Control Baselines

- Added control modes to `adaptive_threshold_predictor/train_gate.py`:
  - `--control_mode condition_only`: use only normalized timestep and normalized PSNR, no latent-derived feature branch.
  - `--control_mode noise_feature`: keep the cached-feature architecture but replace the feature tensor with random noise.
- Added `ConditionOnlyAdaCacheGate` in `adaptive_threshold_predictor/models.py`.
- Ran both controls on the existing `2x2x2` candidate-inverse cache:
  - cache: `/hy-tmp/wan22_adaptive_threshold_feature_cache_candidate_inverse_20260616_012409`
  - output root: `/hy-tmp/wan22_adaptive_threshold_controls_20260616`
  - epochs: `3`
  - batch size: `256`
  - split: same grouped-by-sample-id seed/default as previous runs.
- Control results:
  - `noise_feature`: params `29377`, best epoch `1`, best val loss `0.014648`, best val MAE `0.131173`
  - `condition_only`: params `12865`, best epoch `3`, best val loss `0.014652`, best val MAE `0.128916`
- Comparison with the best real-feature `2x2x2` result:
  - `temporal_mean`: params `29377`, best epoch `2`, best val loss `0.012259`, best val MAE `0.120107`
  - `latent_pool`: params `29377`, best epoch `2`, best val loss `0.012755`, best val MAE `0.116558`
- Current takeaway: timestep+PSNR alone already explains much of the threshold label structure, but the best real latent-derived features improve validation loss by about `13%` to `16%` relative to the no-information controls. Three epochs should not be described as full convergence; train loss kept decreasing while validation loss for real features started rising after epoch 1/2, so current runs are short early-stopping comparisons rather than final converged training.

## 2026-06-16 Adaptive Predictor Progress Review

- Reviewed the adaptive-threshold predictor logs, code, and result roots.
- Current predictor scope is timestep/SeaCache-threshold-only, not full three-cache threshold-combination prediction.
- Main code directory: `adaptive_threshold_predictor/`.
- Main training data root: `/hy-tmp/openvid_100_seacache_trace_data/data`.
- Default dataset mode is `candidate_inverse`: candidate latent + normalized step index + achieved PSNR predicts the SeaCache threshold used by that candidate.
- Current formal cached data:
  - `/hy-tmp/wan22_adaptive_threshold_feature_cache_candidate_inverse_20260616_012409`
  - `50000` examples, `2x2x2` pooled features, five feature sets, feature dim `128`.
- Current best loss-based model setting remains `2x2x2 temporal_mean`:
  - output root: `/hy-tmp/wan22_adaptive_threshold_feature_ablation_cached_20260616_012409`
  - best val loss `0.012259`, val MAE `0.120107`, params `29377`.
- Larger pooling grids and control baselines were reviewed; neither changes the current default recommendation.
- No code or experiment outputs were changed during this review beyond this progress note and the session log.

## 2026-06-16 Adaptive Predictor Report Draft

- Started `reports/report_adaptive_predictor.md`.
- Wrote section `1. 数据准备` only.
- The section covers:
  - OpenVid-100 SeaCache trace data root and flat `data/` symlink layout.
  - 100 samples, 10 SeaCache thresholds, 1000 candidate runs.
  - Main summary/prompt tables and key fields.
  - Baseline/SeaCache artifact path templates.
  - Step trace tensor layout `[16, 12, 60, 104]`, 50 denoising steps, `float16` on disk.
  - `candidate_inverse` training-sample construction.
  - grouped train/validation split: 40000 train examples and 10000 val examples.
  - PSNR/timestep normalization.
  - cached pooled feature data under `/hy-tmp/wan22_adaptive_threshold_feature_cache_candidate_inverse_20260616_012409`.
- User requested writing the report incrementally; next section is not written yet.

Update:

- Added section `2. 网络架构` to `reports/report_adaptive_predictor.md`.
- The section covers:
  - current single-threshold SeaCache/timestep prediction task;
  - raw latent and cached-feature inputs;
  - timestep and PSNR conditioning;
  - two-branch architecture: feature branch + condition branch + prediction head;
  - five latent-derived feature definitions;
  - cached-feature model path;
  - control models: `feature`, `condition_only`, `noise_feature`;
  - current default config: `CachedFeatureAdaCacheGate`, `temporal_mean`, `2x2x2`, hidden dim `64`, SmoothL1Loss, AdamW.
- Revised the architecture section again to explicitly include:
  - input shapes and value ranges;
  - text flow diagram;
  - output shape and value range;
  - parameter counts for default and larger pooling-grid variants.
- Added an architecture SVG figure:
  - `reports/assets/adaptive_predictor_architecture.svg`
  - embedded in `reports/report_adaptive_predictor.md`
  - includes input/output value ranges, shapes, feature/condition branches, fusion, prediction head, and default parameter count.
- Revised the SVG figure to be PPT-clean:
  - intermediate blocks now keep only module names and shape changes;
  - input/output blocks keep value ranges;
  - detailed layer lists were removed from the figure.
- Added section `3. Ablation 结果汇总` to `reports/report_adaptive_predictor.md`.
- Added two result tables:
  - feature training summary for `2x2x2`, including `condition_only` and `noise_feature` controls;
  - pooling-size training summary across `2x2x2`, `2x4x4`, `3x4x4`, and `4x4x4`.
- Tables include parameter count, best epoch, best train loss, best validation loss, best validation MAE, last validation loss, and last validation MAE.

## 2026-06-16 Three-cache Sea-style vs Old Merge Comparison

- Updated `reports/report_three_cache_sea_threshold_grid_prompt01.md`.
- Added section `3.6 与旧 three-cache merge 实验的代表点对比`.
- Compared Sea-style three-cache grid against the old merge grid:
  - Sea-style root: `/hy-tmp/wan22_three_cache_sea_prompt01_50step_45f_480p_20260614_005404`
  - old merge root: `/hy-tmp/work/Wan2.2/experiment_results/wan22_three_cache_threshold_grid_prompt01_50step_45f_480p_20260610_012518`
- Selected representative fastest candidates under PSNR thresholds and special points:
  - highest finite PSNR;
  - PSNR `>=26`, `>=24`, `>=22`, `>=20`, `>=19`, `>=18`, `>=15`;
  - fastest finite candidate.
- Key comparison:
  - Sea-style highest finite PSNR is `37.465 dB` versus old merge `26.954 dB`.
  - Sea-style gives better speed/quality tradeoff in the `22-26 dB` range.
  - Around `18 dB`, old merge is slightly faster while Sea-style is slightly higher quality.
  - In the aggressive `15-16 dB` range, Sea-style reaches higher speedup.

## 2026-06-16 Adaptive Predictor Promotion To Top-level Package

- Removed tracked image `framework.png` at the user's request.
- Moved adaptive predictor code from `experiments/adaptive_threshold_predictor/` to top-level `adaptive_threshold_predictor/`.
- Updated Python imports and module commands from:
  - `experiments.adaptive_threshold_predictor.*`
  - to `adaptive_threshold_predictor.*`
- Updated `adaptive_threshold_predictor/README.md` to describe the package as top-level adaptive threshold work.
- Removed generated `__pycache__` files during the move.

## 2026-06-16 Adaptive Predictor Long Feature Training

- Reran the two main adaptive predictor feature settings for longer training to check whether the epoch-2 best loss was a short-run fluctuation.
- Command root:
  - `/hy-tmp/wan22_adaptive_threshold_feature_ablation_long_20260616`
- Full stdout log:
  - `logs/2026-06-16_adaptive_feature_ablation_long_train.log`
- Configuration matched the earlier short ablation except for longer training:
  - `candidate_inverse`
  - cached features from `/hy-tmp/wan22_adaptive_threshold_feature_cache_candidate_inverse_20260616_012409`
  - feature sets: `temporal_mean`, `latent_pool`
  - `2x2x2` pooled feature dim `128`
  - hidden dim `64`
  - batch size `256`
  - split seed `42`
  - train/val examples: `40000/10000`
  - epochs: `30`
- Results:
  - `temporal_mean`: best val loss moved to epoch `1`, val loss `0.012571`, val MAE `0.119452`; best val MAE at epoch `2`, val MAE `0.119011`; epoch-30 val loss `0.019334`, val MAE `0.143876`.
  - `latent_pool`: best val loss at epoch `3`, val loss `0.012612`, val MAE `0.121387`; best val MAE at epoch `2`, val MAE `0.117695`; epoch-30 val loss `0.023170`, val MAE `0.155156`.
- Takeaway:
  - The longer curves do not indicate late convergence after epoch 2. Training loss keeps decreasing, while validation loss rises after the first few epochs for both settings, so the earlier best-at-epoch-2 behavior is best interpreted as early overfitting / early-stopping behavior rather than a model that needed more epochs to converge.
  - `temporal_mean` remains the safer loss-based default; `latent_pool` still has slightly better early best MAE but worse validation-loss stability.

## 2026-06-16 Adaptive Predictor Hidden Dim 16 Test

- Reran the same two feature settings with a smaller predictor capacity:
  - output root: `/hy-tmp/wan22_adaptive_threshold_feature_ablation_hdim16_20260616`
  - full stdout log: `logs/2026-06-16_adaptive_feature_ablation_hdim16_train.log`
  - `hidden_dim`: `16`
  - parameters: `3505`
  - epochs: `30`
  - all other data/split/cache settings matched the `hidden_dim=64` long run.
- Best-checkpoint results:
  - `temporal_mean`: best val loss epoch `4`, val loss `0.012254`, val MAE `0.119388`; best val MAE epoch `3`, val MAE `0.119104`; epoch-30 val loss `0.021120`, val MAE `0.151698`.
  - `latent_pool`: best val loss epoch `4`, val loss `0.012473`, val MAE `0.121571`; best val MAE epoch `3`, val MAE `0.118758`; epoch-30 val loss `0.018201`, val MAE `0.144731`.
- Comparison to `hidden_dim=64` long run:
  - `temporal_mean` best val loss improved from `0.012571` to `0.012254`.
  - `latent_pool` best val loss improved from `0.012612` to `0.012473`.
  - `latent_pool` epoch-30 val loss improved from `0.023170` to `0.018201`, so reducing capacity reduced late overfitting for this feature.
  - Early overfitting still remains: validation loss improves for the first few epochs, then rises while training loss continues decreasing.
- Takeaway:
  - Capacity was likely too high for the current grouped-sample supervision. `hidden_dim=16` is a better short-term default candidate than `hidden_dim=64`, but it still needs early stopping and ideally multi-seed validation.

## 2026-06-16 Adaptive Predictor Hidden Dim 8 Test

- Reran the same two feature settings with an even smaller predictor:
  - output root: `/hy-tmp/wan22_adaptive_threshold_feature_ablation_hdim8_20260616`
  - full stdout log: `logs/2026-06-16_adaptive_feature_ablation_hdim8_train.log`
  - `hidden_dim`: `8`
  - parameters: `1433`
  - epochs: `30`
  - all other data/split/cache settings matched the `hidden_dim=64` and `hidden_dim=16` long runs.
- Best-checkpoint results:
  - `temporal_mean`: best val loss epoch `6`, val loss `0.013240`, val MAE `0.122773`; best val MAE epoch `8`, val MAE `0.121107`; epoch-30 val loss `0.019828`, val MAE `0.144768`.
  - `latent_pool`: best val loss epoch `4`, val loss `0.013039`, val MAE `0.120766`; epoch-30 val loss `0.016017`, val MAE `0.132199`.
- Three-capacity comparison:
  - `hidden_dim=64`: best val loss `0.012571` (`temporal_mean`), `0.012612` (`latent_pool`); params `29377`.
  - `hidden_dim=16`: best val loss `0.012254` (`temporal_mean`), `0.012473` (`latent_pool`); params `3505`.
  - `hidden_dim=8`: best val loss `0.013240` (`temporal_mean`), `0.013039` (`latent_pool`); params `1433`.
- Takeaway:
  - `hidden_dim=8` reduces late overfitting, especially for `latent_pool`, but best validation loss gets worse than `hidden_dim=16`.
  - Current capacity sweet spot among tested values is `hidden_dim=16`; `hidden_dim=8` looks under-capacity for best loss, while `hidden_dim=64` overfits faster.

## 2026-06-16 Adaptive Predictor Condition-only Comparison

- Reran condition-only controls for the same long-training setup and hidden dimensions:
  - roots:
    - `/hy-tmp/wan22_adaptive_threshold_condition_only_hdim64_20260616`
    - `/hy-tmp/wan22_adaptive_threshold_condition_only_hdim16_20260616`
    - `/hy-tmp/wan22_adaptive_threshold_condition_only_hdim8_20260616`
  - stdout logs:
    - `logs/2026-06-16_adaptive_condition_only_hdim64_train.log`
    - `logs/2026-06-16_adaptive_condition_only_hdim16_train.log`
    - `logs/2026-06-16_adaptive_condition_only_hdim8_train.log`
  - same cached dataset/split, `30` epochs, batch size `256`.
  - condition-only model uses timestep and target PSNR only; cached feature loading is only used to reuse the same dataset metadata/split.
- Best validation results:
  - `condition_only`, hdim `64`: params `12865`, best val loss `0.013834`, best val MAE `0.125093`.
  - `condition_only`, hdim `16`: params `913`, best val loss `0.014025`, best val MAE `0.125755`.
  - `condition_only`, hdim `8`: params `265`, best val loss `0.014505`, best val MAE `0.128548`.
- Feature-model improvement over same-hidden-dim condition-only:
  - hdim `64`, `temporal_mean`: best val loss improves `9.12%`; best MAE improves `4.86%`.
  - hdim `64`, `latent_pool`: best val loss improves `8.83%`; best MAE improves `5.91%`.
  - hdim `16`, `temporal_mean`: best val loss improves `12.63%`; best MAE improves `5.29%`.
  - hdim `16`, `latent_pool`: best val loss improves `11.07%`; best MAE improves `5.56%`.
  - hdim `8`, `temporal_mean`: best val loss improves `8.72%`; best MAE improves `5.79%`.
  - hdim `8`, `latent_pool`: best val loss improves `10.11%`; best MAE improves `6.05%`.
- Takeaway:
  - Latent-derived features do add real validation signal beyond timestep + target PSNR, with consistent best-checkpoint gains across capacities.
  - Condition-only is more stable late in training, while feature models overfit earlier. Use feature models with early stopping, and compare best checkpoint rather than last epoch.
  - Current strongest single-split setting remains hdim `16` + `temporal_mean` by best val loss.

## 2026-06-16 Adaptive SeaCache Inference Prototype

- Added standalone adaptive SeaCache inference prototype under `adaptive_seacache_wan22/`.
- No main Wan files were modified.
- New files:
  - `adaptive_seacache_wan22/__init__.py`
  - `adaptive_seacache_wan22/cache.py`
  - `adaptive_seacache_wan22/patch.py`
  - `adaptive_seacache_wan22/generate_t2v.py`
  - `adaptive_seacache_wan22/README.md`
- Intended mode:
  - T2V timestep-only SeaCache.
  - User supplies `--target_psnr`.
  - At each SeaCache decision, the adaptive predictor predicts a threshold from the current raw latent and denoising step fraction.
  - SeaCache uses that threshold for the current step only.
- Implementation details:
  - The script monkey-patches `wan.text2video.SeaCacheTimestepCache` in the current process to construct `AdaptiveSeaCacheTimestepCache`.
  - It wraps `WanModel.forward` in the current process so the adaptive cache can access the current raw latent. This is needed because the native SeaCache hook receives model-internal token features, while the adaptive predictor was trained from raw latent pooled features.
  - Online feature extraction mirrors `adaptive_threshold_predictor/build_feature_cache.py` for `temporal_mean` and `latent_pool`.
  - The default recommended model is `/hy-tmp/wan22_adaptive_threshold_feature_ablation_hdim16_20260616/temporal_mean/best_model.pt`.
- Validation:
  - `python -m py_compile adaptive_seacache_wan22/cache.py adaptive_seacache_wan22/patch.py adaptive_seacache_wan22/generate_t2v.py` passed.
  - Lightweight smoke loaded the hdim16 temporal_mean checkpoint and predicted thresholds from fake raw latents.
  - Smoke confirmed `AdaptiveSeaCacheTimestepCache.summary()` records `adaptive_threshold_path`.
  - CLI parse smoke passed for a realistic 832x480/45f/50step T2V command.
- Real Wan generation was not launched yet in this session. First real test should use the command in `adaptive_seacache_wan22/README.md`, ideally with `target_psnr=25` and baseline/SeaCache comparison artifacts recorded.

## 2026-06-16 Adaptive SeaCache Ali Prompt 1-2 Run

- Added batch runner:
  - `experiments/adaptive_seacache_ali_prompt12_50step_45f_480p/run_batch.py`
  - `experiments/adaptive_seacache_ali_prompt12_50step_45f_480p/run_tmux.sh`
  - `experiments/adaptive_seacache_ali_prompt12_50step_45f_480p/README.md`
- Runner scope:
  - prompts: Ali prompt 1 and 2 from `test_sets/ali_10/prompts.txt`
  - targets: `20`, `25`, `30`
  - method: timestep-only adaptive SeaCache
  - model: hdim16 `temporal_mean` best checkpoint
  - single-process WanT2V pipeline load
  - baseline artifacts reused from `/hy-tmp/wan22_zeus_threshold_reuse_interp_10prompt_5th_20260608_195427`
- Trace archiving:
  - per-candidate JSON and CSV under `traces/target_<value>/prompt_<NN>.*`
  - fields include `step_index`, `model_stage`, `branch`, `predicted_threshold`, `rel_l1`, `accumulated_rel_l1`, `decision`, and `force_recompute`.
- Validation before launch:
  - `py_compile` passed for the adaptive cache and runner.
  - CPU validation passed: `2` prompts, `3` target PSNRs, `6` candidate runs, baseline artifacts present.
- Launched tmux session:
  - session: `adaptive_seacache_ali12`
  - experiment root: `/hy-tmp/wan22_adaptive_seacache_ali_prompt12_50step_45f_480p_20260616_165412`
  - runner log: `/hy-tmp/wan22_adaptive_seacache_ali_prompt12_50step_45f_480p_20260616_165412/logs/runner.log`
- Initial log check showed the adaptive gate loaded successfully. The run was still in progress at this checkpoint.

Completion update:

- The tmux session finished and exited.
- Completed `6/6` adaptive SeaCache candidates.
- Failed files: `0`.
- Result table:
  - `/hy-tmp/wan22_adaptive_seacache_ali_prompt12_50step_45f_480p_20260616_165412/results/summary.csv`
  - `/hy-tmp/wan22_adaptive_seacache_ali_prompt12_50step_45f_480p_20260616_165412/results/summary.json`
- All generated and baseline videos validated as `832x480`, `45` frames, duration `2.8125s`, avg frame rate `16/1`.
- Summary:
  - `ali_001`, target `20`: speedup `2.870x`, PSNR `19.325 dB`, reuse/recompute `72/28`, mean predicted threshold `0.5330`.
  - `ali_001`, target `25`: speedup `1.869x`, PSNR `19.450 dB`, reuse/recompute `52/48`, mean predicted threshold `0.2819`.
  - `ali_001`, target `30`: speedup `1.543x`, PSNR `24.462 dB`, reuse/recompute `40/60`, mean predicted threshold `0.1809`.
  - `ali_002`, target `20`: speedup `3.051x`, PSNR `20.288 dB`, reuse/recompute `74/26`, mean predicted threshold `0.6230`.
  - `ali_002`, target `25`: speedup `2.270x`, PSNR `26.998 dB`, reuse/recompute `62/38`, mean predicted threshold `0.3777`.
  - `ali_002`, target `30`: speedup `1.641x`, PSNR `29.354 dB`, reuse/recompute `44/56`, mean predicted threshold `0.2126`.

Fixed SeaCache comparison update:

- Compared adaptive results against prior fixed-threshold SeaCache runs:
  - prompt 01 fixed root: `/hy-tmp/wan22_seacache_50step_45f_480p_20260611_191733`
  - prompt 02 fixed roots:
    - `/hy-tmp/wan22_seacache_prompt02_dense_20260611_204826`
    - `/hy-tmp/wan22_seacache_prompt02_highthr_20260612_000218`
- Nearest fixed-threshold comparisons:
  - `ali_001`, target `20`: adaptive `19.325 dB`, `2.870x`; nearest fixed is threshold `0.50`, `19.460 dB`, `2.779x`.
  - `ali_001`, target `25`: adaptive `19.450 dB`, `1.869x`; nearest PSNR fixed is threshold `0.50`, `19.460 dB`, `2.779x`; nearest speed fixed is threshold `0.30`, `20.562 dB`, `1.966x`. This adaptive point is dominated by fixed SeaCache.
  - `ali_001`, target `30`: adaptive `24.462 dB`, `1.543x`; nearest fixed is threshold `0.20`, `24.558 dB`, `1.569x`.
  - `ali_002`, target `20`: adaptive `20.288 dB`, `3.051x`; nearest fixed is threshold `0.60`, `20.262 dB`, `3.098x`.
  - `ali_002`, target `25`: adaptive `26.998 dB`, `2.270x`; nearest fixed is threshold `0.40`, `27.044 dB`, `2.405x`.
  - `ali_002`, target `30`: adaptive `29.354 dB`, `1.641x`; nearest PSNR fixed is threshold `0.30`, `29.582 dB`, `1.965x`; nearest speed fixed is threshold `0.20`, `30.097 dB`, `1.562x`.
- Takeaway:
  - Adaptive thresholding mostly lands near fixed-threshold SeaCache operating points, which confirms the predicted thresholds drive SeaCache in the expected direction.
  - It does not yet dominate fixed threshold sweeps. The most concerning point is `ali_001 target=25`, where adaptive is much slower than fixed threshold `0.50` at nearly identical PSNR and worse than fixed threshold `0.30` in both speed and PSNR.
  - Prompt 02 is better calibrated than prompt 01, but fixed threshold `0.40`/`0.60` remain slightly better at comparable quality/speed.

## 2026-06-16 AdaCache-DiT Method Review

- Cloned and reviewed the official AdaCache-DiT/AdaCache repository under `/hy-tmp/work/AdaCache`.
- Read `README.md`, `configs/sample_adacache.py`, `configs/sample_adacache_moreg.py`, `inference.py`, and the core implementation in `opensora_base/opensora/models/stdit/stdit3.py`.
- No Wan2.2 implementation changes were made in this pass.
- Key findings:
  - AdaCache is a training-free adaptive residual cache for video DiTs.
  - The released Open-Sora implementation caches selected block residual components (`t-attn`, `s-attn`, or `ca-mlp`) rather than whole denoiser output.
  - It uses a per-sample/per-run codebook mapping residual-change magnitude to the next cache interval, so the recompute cadence adapts during sampling.
  - The default released config uses temporal-attention residual caching at block `13` with a 100-step codebook `{0.03: 12, 0.05: 10, 0.07: 8, 0.09: 6, 0.11: 4, 1.00: 3}`.
  - MoReg multiplies the residual-change metric by a motion regularizer derived from temporal residual differences, allocating more compute to high-motion or rapidly changing motion regions/steps.

## 2026-06-16 AdaCache Wan2.2 Isolated Adapter

- Copied the official AdaCache repository into the Wan2.2 worktree at `third_party/AdaCache` and removed nested `.git` metadata so it can be versioned by this repository.
- Per user instruction, did not modify Wan2.2 main source files (`generate.py`, `wan/`, etc.).
- Added an isolated Wan2.2 runtime adapter under `third_party/AdaCache/wan22_adacache/`:
  - `adapter.py`: monkey-patches `WanModel.forward` and `WanAttentionBlock.forward` at runtime only.
  - `README.md`: documents usage and method mapping.
  - `run_wan22_adacache.py`: wrapper launcher that consumes `--block_cache adacache` and AdaCache-specific arguments, enables the runtime patch, and delegates to Wan2.2 `generate.py`.
- Adapter behavior:
  - caches residuals for every Wan2.2 transformer block, matching the official AdaCache style;
  - uses `cache_loc` only for computing the shared adaptive cadence metric;
  - defaults to the official codebook `0.03:12,0.05:10,0.07:8,0.09:6,0.11:4,1.0:3`;
  - maps `t-attn`/`s-attn`/`self-attn` to Wan2.2 self-attention residuals and `ca-mlp` to cross-attention plus FFN residuals;
  - keys state by explicit `(model_stage, branch)`;
  - clears the completed high/low stage so the next model stage cold-starts.
- Validation run:
  - `py_compile` passed for the new adapter and wrapper using `/hy-tmp/miniconda3/envs/Wan2.2/bin/python`.
  - Wrapper `--help` successfully delegated to Wan2.2 `generate.py`.
- No GPU inference smoke test has been run yet.

## 2026-06-16 AdaCache VBench Batch Runner

- Added `experiments/adacache_vbench_50step_45f_480p/`.
- New files:
  - `run_batch.py`
  - `run_tmux.sh`
  - `README.md`
- Runner design:
  - single-process batch runner;
  - loads WanT2V once;
  - runs selected VBench prompts from `test_sets/vbench_every20/prompts.jsonl`;
  - runs no-cache baseline and AdaCache candidate for each selected prompt;
  - uses project defaults: `t2v-A14B`, `/hy-tmp/models/Wan2.2-T2V-A14B`, seed `42`, `832*480`, `45` frames, `50` steps, `dpm++`, `--offload_model`, dtype conversion enabled;
  - archives videos, ffprobe JSON, PSNR JSON/logs, command records, raw logs, manifests, failed records, `experiment_config.json`, `launch.env`, and summary CSV/JSON under `/hy-tmp`.
- Important adapter fix:
  - added an `enabled` switch to the AdaCache runtime so the runner can keep the monkey patch installed but disabled during baseline runs;
  - candidate runs enable AdaCache only for the candidate generation and clear state afterward.
- Validation:
  - `py_compile` passed for `third_party/AdaCache/wan22_adacache/adapter.py` and the new runner.
  - CPU validate passed for both full VBench prompt set (`51` prompts) and a two-prompt subset.
- GPU smoke test launched afterward; see next entry.

## 2026-06-16 AdaCache VBench Smoke OOM

- Ran one-prompt VBench smoke test:
  - command: `experiments/adacache_vbench_50step_45f_480p/run_batch.py --prompt_limit 1 --exp_root /hy-tmp/wan22_adacache_vbench_smoke_20260616_1908 --convert_model_dtype`
  - prompt: `vbench_every20_001`
  - baseline + AdaCache candidate, default AdaCache config (`t-attn`, `cache_loc=13`, official codebook, no MoReg).
- GPU mode was confirmed before launch:
  - `NVIDIA A100 80GB PCIe`
  - initially `0 MiB / 81920 MiB` used.
- Baseline completed successfully:
  - video: `/hy-tmp/wan22_adacache_vbench_smoke_20260616_1908/baseline/vbench_every20_001.mp4`
  - ffprobe JSON exists.
  - compute elapsed: `533.455s`.
  - observed baseline sampling memory was about `44.0GB`.
- AdaCache candidate OOMed at step `0`, during the `uncond` branch self-attention call:
  - log: `/hy-tmp/wan22_adacache_vbench_smoke_20260616_1908/logs/adacache_vbench_every20_001.log`
  - failed record: `/hy-tmp/wan22_adacache_vbench_smoke_20260616_1908/failed/adacache_vbench_every20_001.txt`
  - error: tried to allocate `732 MiB`; only `716.94 MiB` free.
  - process had `78.54 GiB` in use; PyTorch allocated `73.15 GiB` and reserved `4.90 GiB`.
  - observed `nvidia-smi` just before failure showed about `81001 MiB / 81920 MiB`.
- Takeaway:
  - The fully official-style AdaCache implementation that caches residuals for all Wan2.2 blocks and keeps explicit `cond`/`uncond` branch state does not fit this single A100 80GB setup at the default `832*480`, `45f`, `50-step` configuration.
  - A reduced-memory variant is needed for practical comparison, e.g. selected-block-only caching, CPU/offloaded cache tensors, or avoiding simultaneous cond/uncond full-block residual caches.

## 2026-06-17 Multi-session Commit Checkpoint

- User requested committing accumulated progress from multiple sessions.
- Reviewed working tree before staging:
  - `PROGRESS.md` contains accumulated entries for TaylorSeer third-party move, VBench runners, adaptive predictor training comparisons, adaptive SeaCache prototype/run, AdaCache method review/adapter/runner/smoke OOM.
  - top-level `taylorseer_wan22/` was deleted after moving the standalone integration to `third_party/taylorseer_wan22/`.
  - new code/directories include `adaptive_seacache_wan22/`, `experiments/adacache_vbench_50step_45f_480p/`, `experiments/adaptive_seacache_ali_prompt12_50step_45f_480p/`, `experiments/taylorseer_vbench_50step_45f_480p/`, and `third_party/`.
  - new logs under `logs/2026-06-16_*` document implementation notes, training logs, launches, and OOM diagnosis.
- Size check:
  - `third_party/` is about `301M`, mainly the copied AdaCache/Open-Sora source tree and included demo assets; no file larger than `20M` was found in the new code/directories checked.
  - ignored `__pycache__/` files were not staged.
- No new validation was run during this commit-only session; validation status is the one recorded in the individual 2026-06-16 progress entries.

## 2026-06-23 Adaptive SeaCache Cache Lifecycle OOM Fix

- Fixed the adaptive SeaCache batch-runner OOM root cause: the factory kept a historical `instances` list of adaptive/replay SeaCache cache objects. Those objects retain GPU tensors in runtime state, including SeaCache `previous_feature`/`previous_residual` and adaptive current-latent snapshots, so `torch.cuda.empty_cache()` could not free memory while old cache objects were still referenced.
- Updated `adaptive_seacache_wan22/cache.py`:
  - removed historical cache-instance retention from adaptive and replay factories;
  - added `clear_runtime_state()` on adaptive and replay cache classes;
  - added factory `clear_last_instance()` hooks that clear runtime state and drop the latest cache reference after trace/summary extraction.
- Updated adaptive batch runners to release cache state after writing per-candidate traces and also on exception paths:
  - `experiments/adaptive_seacache_ali_prompt12_50step_45f_480p/run_batch.py`
  - `experiments/adaptive_seacache_train10_50step_45f_480p/run_batch.py`
  - `experiments/adaptive_seacache_train15_test5_50step_45f_480p/run_batch.py`
  - `experiments/adaptive_seacache_overhead_train5_50step_45f_480p/run_batch.py`
- Documentation/spec updates:
  - `AGENTS.md` now explicitly states that single-process batch runners must not retain historical SeaCache/adaptive SeaCache/replay SeaCache instances and must release cache runtime state after each candidate.
  - Adaptive prototype and experiment READMEs now call out the cache lifecycle requirement.
  - The overhead experiment README was corrected to describe the train5 online-vs-replay overhead experiment instead of the copied train15/test5 text.
- Validation:
  - `python -m py_compile` passed for `adaptive_seacache_wan22/cache.py` and all four adaptive SeaCache runners.
  - `bash -n` passed for the four adaptive SeaCache `run_tmux.sh` launch scripts.
  - `rg` confirmed no remaining `self.instances` / `instances.append` retention in the adaptive SeaCache cache module or runners.

## 2026-06-23 VBench10 OSS Result/Report Import

- User requested importing two OSS handoff packages from `oss://datasets/`:
  - `wan22_vbench10_reports_and_experiment_scripts_20260623.tar.gz`
  - `wan22_vbench10_three_cache_full_merge_and_timestep_only_full_20260623.tar.gz`
- Logged into OSS with the provided Hengyuan account and downloaded both packages plus checksum archives to `/hy-tmp/oss_downloads_20260623/`.
- SHA256 verification passed:
  - reports/scripts package: `a95282ae6e567e69b033ffcbc46abb657a4a8f7feaaee9fd87e232e20612b9bc`
  - full results package: `1211cb89b75b75b340a9e53910db52f9e1c981b1e12694b400fc66ce03be4a84`
- Imported reports into `reports/`:
  - `reports/report_three_cache_sea_vbench10_merge.md`
  - `reports/report_timestep_only_seacache_vbench10.md`
  - `reports/report_compare_three_cache_merge_vs_timestep_only_vbench10.md`
- Imported experiment scripts into `experiments/`:
  - `experiments/seacache_vbench10_50step_45f_480p/`
  - `experiments/three_cache_sea_vbench10_50step_45f_480p/`
- Extracted full runtime results to:
  - `/hy-tmp/wan22_vbench10_three_cache_full_merge_and_timestep_only_full_20260623/`
- Added workspace symlink:
  - `experiment_results/wan22_vbench10_three_cache_full_merge_and_timestep_only_full_20260623 -> /hy-tmp/wan22_vbench10_three_cache_full_merge_and_timestep_only_full_20260623`
- Result package contents:
  - `three_cache_sea_vbench10_full/`: full VBench10 three-cache Sea-style result set with `shard_gpu0_p000_004`, `shard_gpu1_p005_009`, and merged summary/aggregate tables.
  - `timestep_only_seacache_vbench10_full/`: full VBench10 timestep-only SeaCache result set from `/hy-tmp/wan22_seacache_vbench10_50step_45f_480p_20260618_161845`, including shard outputs, merged tables, videos, logs, ffprobe, and PSNR artifacts.
- Lightweight integrity check after extraction:
  - `760` mp4 files
  - `26` csv files
  - `1532` json files
  - `0` files under `failed/`
- Disk note: `/hy-tmp` had about `69G` free after extraction. Downloaded tarballs remain in `/hy-tmp/oss_downloads_20260623/` and can be removed later if space is needed.

## 2026-06-23 AdaCache VBench10 Reproduction OSS Import

- User requested importing the full AdaCache reproduction archive from:
  - `oss://datasets/adacache_reproduction_20260623/adacache_wan22_vbench10_reproduction_20260623.tar.gz`
- Downloaded the archive to:
  - `/hy-tmp/oss_downloads_20260623/adacache_wan22_vbench10_reproduction_20260623.tar.gz`
- No separate OSS checksum archive was listed for this object; local SHA256 was recorded:
  - `db66e61a1180f1c91e0b3b0643bdd07baf813b3c8dbb2a293164a61932b7b7a5`
- Extracted the full archive to:
  - `/hy-tmp/adacache_wan22_vbench10_reproduction_20260623/`
- Added workspace symlink:
  - `experiment_results/adacache_wan22_vbench10_reproduction_20260623 -> /hy-tmp/adacache_wan22_vbench10_reproduction_20260623`
- Imported report into `reports/`:
  - `reports/adacache_vbench10_reproduction_report.md`
- Imported experiment scripts into:
  - `experiments/adacache_wan22_vbench10_reproduction_20260623/`
  - imported files: `README.md`, `run_batch.py`, `run_one_method.py`, `run_tmux.sh`
  - `__pycache__/` and `*.pyc` were excluded from the repository copy.
- Result package contents:
  - `experiment_results/baseline/`: 10 no-cache VBench10 baseline videos.
  - `experiment_results/slow/`: 10 AdaCache slow videos.
  - `experiment_results/fast/`: 10 AdaCache fast videos.
  - `experiment_results/results/summary_all.csv`
  - `experiment_results/results/aggregate_all.json`
  - per-shard summaries, commands, logs, ffprobe artifacts, PSNR artifacts, manifests, launch envs, and GPU/runtime records.
- Lightweight integrity check after extraction:
  - `30` mp4 files
  - `35` csv files
  - `85` json files
  - `0` files under `failed/`
- Archived report summary:
  - AdaCache slow: mean speedup `1.545x`, mean PSNR `23.561 dB`
  - AdaCache fast: mean speedup `2.702x`, mean PSNR `18.635 dB`
  - completion: 10/10 baseline videos, 10/10 slow videos, 10/10 fast videos, 20/20 PSNR JSON files, no failed samples.

## 2026-06-23 Fixed SeaCache Train15/Test5 OpenVid20 Launch

- User requested a timestep-only fixed-threshold SeaCache control on the same 20 OpenVid prompts used by the adaptive SeaCache train15/test5 report.
- Thresholds selected for the control run: `0.1`, `0.2`, `0.4`, `0.6`.
- Added experiment runner:
  - `experiments/seacache_train15_test5_50step_45f_480p/run_batch.py`
  - `experiments/seacache_train15_test5_50step_45f_480p/run_tmux.sh`
  - `experiments/seacache_train15_test5_50step_45f_480p/README.md`
- Runner behavior:
  - reuses no-cache OpenVid baselines from `experiment_results/openvid_100_seacache_trace_data`;
  - uses the same predictor split JSON and random seed `20260619` as adaptive train15/test5;
  - loads WanT2V once and runs 80 candidates: 20 prompts x 4 thresholds;
  - archives videos, commands, logs, ffprobe JSON, FFmpeg PSNR JSON/logs, per-candidate summary, aggregate-by-threshold table, and failed records.
- Validation before launch:
  - GPU confirmed: `NVIDIA A100 80GB PCIe`, 81920 MiB, driver 570.211.01.
  - `python -m py_compile experiments/seacache_train15_test5_50step_45f_480p/run_batch.py` passed.
  - `run_batch.py --cpu_validate --thresholds '0.1 0.2 0.4 0.6'` passed with 20 prompts, 15 train / 5 test, 80 expected candidates, and no missing baseline artifacts.
  - `bash -n experiments/seacache_train15_test5_50step_45f_480p/run_tmux.sh` passed.
- Launched tmux session:
  - session: `seacache_train15_test5`
  - result root: `/hy-tmp/wan22_seacache_train15_test5_50step_45f_480p_20260623_160513`
  - workspace symlink: `experiment_results/wan22_seacache_train15_test5_50step_45f_480p_20260623_160513`
- Initial runtime check:
  - runner reached fixed-threshold SeaCache sampling for `openvidhd_part1_085`, threshold `0.1`;
  - GPU memory was about `47575 MiB` with `100%` utilization;
  - no failed files and no completed mp4 files at the first sampling check.

## 2026-06-24 Fixed SeaCache Train15/Test5 OpenVid20 Completed

- Fixed-threshold timestep-only SeaCache control completed successfully.
- Result root:
  - `/hy-tmp/wan22_seacache_train15_test5_50step_45f_480p_20260623_160513`
- Workspace symlink:
  - `experiment_results/wan22_seacache_train15_test5_50step_45f_480p_20260623_160513`
- Completion/integrity:
  - candidate mp4 files: `80/80`
  - PSNR JSON files: `80/80`
  - ffprobe candidate JSON files: `80/80`
  - command records: `80/80`
  - failed files: `0`
  - result tables written under `results/`: `summary.csv`, `summary.json`, `aggregate_by_threshold.csv`, `aggregate_by_threshold.json`
- Aggregate results across all 20 prompts:
  - threshold `0.1`: overall speedup `1.138x`, mean PSNR `42.861 dB`, min PSNR `34.33 dB`
  - threshold `0.2`: overall speedup `1.607x`, mean PSNR `30.548 dB`, min PSNR `16.88 dB`
  - threshold `0.4`: overall speedup `2.467x`, mean PSNR `23.936 dB`, min PSNR `15.93 dB`
  - threshold `0.6`: overall speedup `3.176x`, mean PSNR `21.229 dB`, min PSNR `15.04 dB`
- Split means:
  - train threshold `0.1/0.2/0.4/0.6`: mean PSNR `42.483/30.821/23.794/20.634 dB`, mean speedup `1.137/1.608/2.468/3.178x`
  - test threshold `0.1/0.2/0.4/0.6`: mean PSNR `43.993/29.726/24.361/23.014 dB`, mean speedup `1.143/1.606/2.464/3.173x`
- Immediate comparison against the adaptive SeaCache train15/test5 report:
  - fixed SeaCache `0.6` is close to adaptive target-20 speed and slightly higher mean PSNR: `3.176x`, `21.229 dB` vs adaptive target-20 `3.171x`, `21.108 dB`
  - fixed SeaCache `0.4` is close to adaptive target-25 speed and slightly higher mean PSNR: `2.467x`, `23.936 dB` vs adaptive target-25 `2.461x`, `23.407 dB`
  - fixed SeaCache `0.2` has much higher mean PSNR but lower speed than adaptive target-30: `1.607x`, `30.548 dB` vs adaptive target-30 `1.904x`, `27.221 dB`
- Note: another tmux session, `wan22_zeus_vbench10_20260624_003030`, is currently active and using the GPU; this is unrelated to the completed fixed SeaCache control run.

## 2026-06-24 Fixed vs Adaptive SeaCache OpenVid20 Pareto Charts

- User requested charts matching the style of `reports/assets/vbench10_three_cache/`.
- Added plotting script:
  - `reports/plot_seacache_adaptive_train15_test5.py`
- Generated chart assets under:
  - `reports/assets/seacache_adaptive_train15_test5/`
- Generated three chart sets, each in PNG/PDF/SVG:
  - `openvid20_fixed_seacache_pareto_scatter`
  - `openvid20_adaptive_seacache_pareto_scatter`
  - `openvid20_fixed_vs_adaptive_seacache_pareto_overlay`
- Also wrote plot aggregate/source metadata:
  - `openvid20_fixed_seacache_plot_aggregate.csv`
  - `openvid20_adaptive_seacache_plot_aggregate.csv`
  - `plot_inputs.json`
- Validation:
  - `python -m py_compile reports/plot_seacache_adaptive_train15_test5.py` passed.
  - Script ran successfully with conda env Python.
  - Visual inspection of the overlay PNG passed after shortening the source note to avoid an overly wide bbox.
- Important metric note:
  - Plot aggregate speedup uses total baseline compute time divided by total candidate compute time for each method/setting. This matches the fixed SeaCache aggregate table and differs slightly from the adaptive report's mean-of-per-prompt speedup values.

## 2026-06-24 Train-Split Fixed vs Adaptive SeaCache Tables/Chart

- Extended `reports/plot_seacache_adaptive_train15_test5.py` to emit train-split-only result tables and an overlay Pareto chart.
- New/updated outputs under `reports/assets/seacache_adaptive_train15_test5/`:
  - `openvid20_train_fixed_seacache_results.csv` (`60` rows = 15 train prompts x 4 thresholds)
  - `openvid20_train_adaptive_seacache_results.csv` (`45` rows = 15 train prompts x 3 targets)
  - `openvid20_train_fixed_seacache_aggregate.csv`
  - `openvid20_train_adaptive_seacache_aggregate.csv`
  - `openvid20_train_fixed_vs_adaptive_seacache_aggregate.csv`
  - `openvid20_train_fixed_vs_adaptive_seacache_pareto_overlay.{png,pdf,svg}`
- Train-split aggregate results:
  - fixed threshold `0.1/0.2/0.4/0.6`: speedup `1.137/1.607/2.468/3.177x`, mean PSNR `42.483/30.821/23.794/20.634 dB`
  - adaptive target `20/25/30`: speedup `3.159/2.387/1.840x`, mean PSNR `20.481/23.137/27.477 dB`
- Validation:
  - plotting script ran successfully;
  - `python -m py_compile reports/plot_seacache_adaptive_train15_test5.py` passed;
  - visual inspection of the train overlay passed after correcting the x-axis label to 15 train prompts;
  - temporary `reports/__pycache__/` was removed.

## 2026-06-24 ZEUS VBench10 Launch With Reused Baselines

- Updated `experiments/zeus_vbench10_50step_45f_480p/run_batch.py` and `run_tmux.sh` so VBench10 no-cache baselines are reused by default from:
  - `/hy-tmp/work/Wan2.2/experiment_results/wan22_vbench10_three_cache_full_merge_and_timestep_only_full_20260623`
- Baseline reuse behavior:
  - finds existing `baseline/{sample_id}.mp4`, `logs/baseline_{sample_id}.time`, `ffprobe/baseline_{sample_id}.json`, and baseline log artifacts in the imported VBench10 result package;
  - symlinks those artifacts into the new ZEUS experiment root using the standard local layout;
  - does not regenerate no-cache baselines unless `--baseline_reuse_root ""` is passed directly to `run_batch.py`.
- Validation before launch:
  - `python -m py_compile experiments/zeus_vbench10_50step_45f_480p/run_batch.py experiments/zeus_vbench10_50step_45f_480p/summarize_results.py` passed.
  - `python experiments/zeus_vbench10_50step_45f_480p/run_batch.py --cpu_validate` passed with 10 reusable baselines found, 0 expected baseline generations, 10 fixed-ZEUS candidates, and 50 ZEUS-threshold candidates.
  - `bash -n experiments/zeus_vbench10_50step_45f_480p/run_tmux.sh` passed.
  - GPU was idle before launch: A100 80GB, `0 MiB`, `0%`; no tmux sessions were running.
- Launched full tmux run:
  - session: `wan22_zeus_vbench10_20260624_003030`
  - result root: `/hy-tmp/wan22_zeus_vbench10_50step_45f_480p_20260624_003030`
  - workspace symlink: `experiment_results/wan22_zeus_vbench10_50step_45f_480p_20260624_003030`
  - thresholds: `0.005 0.02 0.08 0.20 0.60`
  - fixed ZEUS config: acc range `8-47`, denominator `3`, modular `0 1`, `reuse_interp`, max interval `6`, lagrange `4/4/24`.
- Initial runtime check:
  - runner successfully reused baseline artifacts for `vbench10_001`;
  - fixed ZEUS generation for `vbench10_001` started;
  - GPU was active at about `44023 MiB` and `100%` utilization;
  - no failed files were present at the first runtime check.

## 2026-06-24 ZEUS VBench10 Result Check

- Checked completed ZEUS VBench10 run:
  - result root: `/hy-tmp/wan22_zeus_vbench10_50step_45f_480p_20260624_003030`
  - workspace symlink: `experiment_results/wan22_zeus_vbench10_50step_45f_480p_20260624_003030`
- Completion status:
  - fixed ZEUS videos/PSNR: `10/10`
  - ZEUS-threshold videos/PSNR: `50/50`
  - result rows: `60`
  - failed files: `0`
  - ffprobe JSON files: `70`; all checked as `832x480`, `45` frames.
- Aggregate results from `results/aggregate_by_method.csv`:
  - fixed ZEUS: `2.021x`, mean PSNR `23.996 dB`, min PSNR `14.96 dB`, timestep reuse/recompute `250/250`.
  - ZEUS-threshold `0.005`: `1.129x`, mean PSNR `23.020 dB`, min PSNR `14.40 dB`, reuse/recompute `50/450`.
  - ZEUS-threshold `0.02`: `1.604x`, mean PSNR `20.868 dB`, min PSNR `14.31 dB`, reuse/recompute `184/316`.
  - ZEUS-threshold `0.08`: `2.282x`, mean PSNR `20.690 dB`, min PSNR `14.87 dB`, reuse/recompute `279/221`.
  - ZEUS-threshold `0.20`: `2.648x`, mean PSNR `20.707 dB`, min PSNR `14.92 dB`, reuse/recompute `310/190`.
  - ZEUS-threshold `0.60`: `2.793x`, mean PSNR `20.734 dB`, min PSNR `14.91 dB`, reuse/recompute `320/180`.
- Current environment note at check time: `nvidia-smi` returned `No devices were found`; results were checked from disk after the tmux run completed.

## 2026-06-25 VS Code Remote Stability Fix

- Investigated repeated VS Code Remote disconnects showing "remote host terminated 3 times within 5 minutes".
- Found recent VS Code Remote logs where Extension Host was repeatedly terminated by `SIGKILL` shortly after startup; system disk, memory, and inode usage were not the cause.
- Added `.vscode/settings.json` to exclude large experiment/model/cache symlinks and disable expensive Python/Pylance indexing/test discovery for this workspace.
- Updated `.gitignore` so only `.vscode/settings.json` is tracked while other `.vscode` local state remains ignored.
- After applying the workspace settings and reconnecting, observed Extension Host staying alive for more than six minutes with CPU reduced from startup spikes to single digits; the earlier 30-60 second `SIGKILL` loop did not recur during the observation window.
- No inference, PSNR, or dataset jobs were run.

## 2026-06-25 A800-2 Merge

- Merged remote branch `x10ngyx/A800-2` into `main`.
- User decision for conflicts: keep both versions of the conflicting ZEUS VBench10 scripts without attempting to fuse them.
- Resolution:
  - Kept the existing `main` versions at `experiments/zeus_vbench10_50step_45f_480p/run_batch.py` and `summarize_results.py`.
  - Saved the A800-2 versions as a separate experiment under `experiments/zeus_unipc_vbench10_50step_45f_480p/`.
- A800-2's only actual modification to a pre-existing main-tracked file was adding `/hy-tmp/env/Wan2.2/bin/ffmpeg` as an FFmpeg fallback in `experiments/zeus_timestep_cache_50step_45f_480p/compute_psnr.py`.
- Validation:
  - `python -m py_compile` passed for the main ZEUS VBench10 scripts, the new ZEUS UniPC VBench10 scripts, and `compute_psnr.py`.
  - `git diff --cached --check` passed.
- No inference, PSNR, or dataset jobs were run.

## 2026-06-25 BUG.md Cache Review

- Reviewed external `BUG.md` claims against current repository code.
- Static review only; no GPU inference, PSNR, or regression experiments were run.
- Main conclusion: many items in `BUG.md` are real code patterns but not clearly correctness bugs. SeaCache `previous_feature` updates on reuse and accumulated distance behavior are consistent with comparing consecutive filtered features / accumulated-threshold gating, while adaptive predictor raw-latent features are intentionally matched between feature-cache training and online inference.
- Items still worth treating as actionable or at least low-risk cleanup candidates: SeaCache/SeaCFG/BlockGroup scheduler sigma anchoring should be rechecked against the intended SeaCache formula; SeaCache default final-step cutoff is a conservative speed tradeoff; FFT gain normalization near zero is a robustness edge case; BlockGroup accumulated cutoff can clear `pending_feature` defensively.
- `BUG.md` remains untracked and was not modified.

## 2026-06-25 Official SeaCache Alignment

- Cloned official SeaCache reference to `/hy-tmp/seacache_official_ref` for code comparison.
- Official reference commit: `3b1c688 Update README.md`.
- Compared official `Wan2.1/seacache_generate.py` and `Wan2.1/util_seacache.py` with local `wan/timestep_cache.py`.
- Confirmed official Wan2.1 SeaCache behavior:
  - state is split per CFG branch in the reference by even/odd call count; local explicit `(model_stage, branch)` keys remain the correct project adaptation.
  - first-block modulated norm input is the metric feature.
  - SEA filter uses `scheduler.sigmas[idx]`, not `idx + 1`.
  - accumulated relative L1 is not reset on reuse; it resets only on forced recompute windows or threshold crossing.
  - default retention/cutoff is first step and final step per branch; with `use_ret_steps`, first 5 steps and no final cutoff.
  - `previous_feature` is updated on every call.
- Fixed one real local mismatch: during retention/cutoff/history-missing/forced recompute paths, local SeaCache now stores the unfiltered modulated input and skips SEA filtering, matching official Wan2.1. Before this change, local code filtered these forced-compute steps before storing `previous_feature`.
- Validation:
  - `python -m py_compile wan/timestep_cache.py` passed.
  - CPU-only behavior check passed by loading `wan/timestep_cache.py` directly and verifying filter calls are skipped for ret/cutoff steps but used for middle reuse decisions.

## 2026-06-27 SeaCache Official Alignment Recheck

- Rechecked local Wan2.2 timestep SeaCache against official SeaCache Wan2.1 reference at `/hy-tmp/seacache_official_ref` commit `3b1c688`.
- Static comparison sources:
  - official: `/hy-tmp/seacache_official_ref/Wan2.1/seacache_generate.py`
  - official: `/hy-tmp/seacache_official_ref/Wan2.1/util_seacache.py`
  - local: `wan/timestep_cache.py`, `wan/modules/model.py`, `wan/text2video.py`, `generate.py`
- Confirmed core behavior is aligned:
  - metric feature is first block's timestep-modulated norm input.
  - filtered middle-step feature uses flow scheduler sigma at the same step index.
  - SEA filter formula and mean-normalized full FFT implementation match official output exactly on a CPU tensor check.
  - threshold gating uses accumulated relative L1 and resets accumulated distance only on threshold-crossing recompute or forced retention/cutoff windows.
  - cache hit reuses the previous transformer-block residual and still runs the model head/unpatchify.
  - default ret/cutoff semantics are equivalent to official first and final denoising step forced recompute per branch; `--seacache_use_ret_steps` maps to first 5 steps and no tail cutoff.
- Necessary Wan2.2 adaptations:
  - local state uses explicit `(model_stage, branch)` keys instead of official Wan2.1's even/odd model-call counter, because Wan2.2 switches high/low DiT stages and also composes with outer CFG cache.
  - local SeaCache is integrated inside `WanModel.forward` instead of monkey-patching class-level model state.
  - local code passes scheduler sigmas into the cache object rather than storing the scheduler on the model class.
  - local code exposes tunable `power_const`, `eps`, `norm_mode`, and optional `num_steps`; official Wan2.1 hard-codes the practical run path to `power_exp=3.0`, `norm_mode=mean`, `eps=1e-16`, `power_const=1.0`.
- Validation:
  - `python -m py_compile wan/timestep_cache.py wan/modules/model.py wan/text2video.py generate.py` passed.
  - `git diff --check` passed.
  - CPU direct-load test confirmed local SEA filter exactly matches official `apply_sea_with_scheduler` for the checked tensor and scheduler sigmas (`filter_max_abs_diff 0.0`).
  - CPU direct-load test confirmed ret/cutoff forced steps do not call `_filter_feature`, while middle steps do.
- No GPU inference, PSNR, or official Ali/Wan2.1 latency reproduction run was launched in this recheck.

## 2026-06-27 SeaCache UniPC Ali-10 / VBench10 Scripts

- User requested checking sampler impact on generation quality by switching the SeaCache experiment solver to `unipc` while keeping other settings unchanged, for `ali_10` and `VBench10`, with separate experiment scripts and queued execution.
- Added separate launch wrappers:
  - `experiments/seacache_unipc_ali10_50step_45f_480p/run_tmux.sh`
  - `experiments/seacache_unipc_vbench10_50step_45f_480p/run_tmux.sh`
  - READMEs in both experiment directories.
- Added queue launcher:
  - `experiments/seacache_unipc_queue_ali10_vbench10_50step_45f_480p.sh`
  - Queue order: Ali-10 first, VBench10 second.
- Both wrappers reuse the existing single-process SeaCache runner `experiments/seacache_vbench10_50step_45f_480p/run_batch.py` and pass `--sample_solver unipc`.
- Experiment settings retained:
  - task `t2v-A14B`, ckpt `/hy-tmp/models/Wan2.2-T2V-A14B`, seed `42`, size `832*480`, frame_num `45`, sample_steps `50`, offload/dtype defaults, timestep cache `seacache`, block cache none, CFG cache none.
  - thresholds `0.10 0.20 0.30 0.50`.
  - output roots default to `/hy-tmp/wan22_seacache_unipc_ali10_50step_45f_480p_<timestamp>` and `/hy-tmp/wan22_seacache_unipc_vbench10_50step_45f_480p_<timestamp>`.
- Validation:
  - `bash -n` passed for both new `run_tmux.sh` wrappers and the queue script.
  - CPU validation passed for Ali-10 after threshold narrowing: 10 prompts, 4 thresholds, expected 10 baselines and 40 SeaCache candidates.
  - CPU validation passed for VBench10 after threshold narrowing: 10 prompts, 4 thresholds, expected 10 baselines and 40 SeaCache candidates.
- Run status:
  - Initially not launched because the instance was not in GPU mode; `nvidia-smi` returned `No devices were found`.
  - Later launched after GPU became available.
  - Queue session: `wan22_seacache_unipc_queue_20260627_023222`.
  - Ali-10 session: `wan22_seacache_unipc_queue_20260627_023222_ali10`.
  - Ali-10 result root: `/hy-tmp/wan22_seacache_unipc_ali10_50step_45f_480p_20260627_023222`.
  - VBench10 result root, queued second: `/hy-tmp/wan22_seacache_unipc_vbench10_50step_45f_480p_20260627_023222`.
  - Launch check: GPU was `NVIDIA A100 80GB PCIe`, idle at launch; Ali-10 runner started and was loading WanT2V checkpoint shards.
  - User asked whether old ZEUS UniPC baselines could be reused. The queue was paused to avoid unnecessary baseline generation.
  - The old ZEUS UniPC reports show matching baseline parameters, but the expected local roots are missing and the `experiment_results/wan22_zeus_unipc_*` symlinks are dangling:
    - `/hy-tmp/wan22_zeus_unipc_ali10_50step_45f_480p_20260624_195011`
    - `/hy-tmp/wan22_zeus_unipc_vbench10_50step_45f_480p_20260624_192306`
  - Downloaded VBench archives contain reusable DPM++ VBench baselines, not UniPC baselines, so they cannot be used as the UniPC PSNR reference.
  - The first launch had already completed `ali_001` baseline artifacts in the Ali-10 result root. The queue was restarted with the same timestamp/root and `--resume_existing`, so `ali_001` can be skipped while the remaining missing baselines/candidates run.
  - Restarted queue session: `wan22_seacache_unipc_queue_20260627_023222`; Ali-10 child session restarted at 2026-06-27 02:54.
  - User pointed out the initial wrappers did not satisfy the separate batch-runner experiment-script requirement because they called the old VBench10 runner directly.
  - Corrected by adding dedicated formal runners and summarizers:
    - `experiments/seacache_unipc_ali10_50step_45f_480p/run_batch.py`
    - `experiments/seacache_unipc_ali10_50step_45f_480p/summarize_results.py`
    - `experiments/seacache_unipc_vbench10_50step_45f_480p/run_batch.py`
    - `experiments/seacache_unipc_vbench10_50step_45f_480p/summarize_results.py`
  - Fixed wrappers to call their own local `run_batch.py`, use `selected_records.{jsonl,csv}`, set `sample_solver=unipc` in summaries, and write method-specific result roots.
  - Validation after correction:
    - `python -m py_compile` passed for both new `run_batch.py` and `summarize_results.py` files.
    - CPU validation passed for both new runners with 10 prompts, 4 thresholds, and 40 expected candidates.
    - `bash -n` and `git diff --check` passed.
  - Restarted the queue again at 2026-06-27 03:14 using the corrected dedicated Ali-10 runner; process command confirmed `experiments/seacache_unipc_ali10_50step_45f_480p/run_batch.py`.
  - Added and launched a periodic monitor:
    - script: `experiments/seacache_unipc_monitor_ali10_vbench10_50step_45f_480p.sh`
    - tmux session: `wan22_seacache_unipc_monitor_20260627_023222`
    - log: `/hy-tmp/wan22_seacache_unipc_ali10_50step_45f_480p_20260627_023222/logs/queue_monitor.log`
    - interval: 600 seconds.
  - First monitor record showed Ali-10 running normally, GPU at about `47351 MiB` and `100%`, 1 baseline MP4, 1 candidate MP4/time/PSNR from resumed partial output, and 0 failed files; VBench10 had not started yet.
  - VBench10 progress check at 2026-06-27 14:41 CST:
    - tmux sessions still active: queue, monitor, and `wan22_seacache_unipc_queue_20260627_023222_vbench10`.
    - GPU active: `NVIDIA A100 80GB PCIe`, about `47345 MiB`, `100%`.
    - VBench10 artifacts: 10/10 baselines, 38/40 SeaCache candidates, 38/40 candidate time files, 38/40 PSNR JSON files, 48/50 ffprobe JSON files, 0 failed files.
    - Missing candidates were only `th_0p30/vbench10_010.mp4` and `th_0p50/vbench10_010.mp4`; runner log showed it was processing `vbench10_010`.
    - Summary/aggregate tables were not yet generated because the batch had not completed.

## 2026-06-27 SeaCache DPM++ Ali-10 Result Check

- User asked whether SeaCache with `dpm++` on full `ali_10` had already been measured.
- Checked `PROGRESS.md`, `reports/`, `logs/`, `experiments/`, `/hy-tmp/wan22_*` result roots, and `experiment_results/` symlinks.
- Finding: no complete formal `SeaCache + dpm++ + ali_10` 10-prompt result root was found.
- Existing related data:
  - `SeaCache + dpm++` on Ali prompt 1 only: `/hy-tmp/wan22_seacache_50step_45f_480p_20260611_191733`.
  - `SeaCache + dpm++` on Ali prompt 2 only: `/hy-tmp/wan22_seacache_prompt02_dense_20260611_204826` and `/hy-tmp/wan22_seacache_prompt02_highthr_20260612_000218`.
  - Full `SeaCache + unipc + ali_10`: `/hy-tmp/wan22_seacache_unipc_ali10_50step_45f_480p_20260627_023222`, 10 baselines and 40 candidates completed.
- Conclusion: use the prompt-01/02 `dpm++` SeaCache results only as a small pilot; a formal full Ali-10 `dpm++` SeaCache run is still missing if required for direct comparison.

## 2026-06-30 Gated 4-Feature MLP Long Row-Split Check

- User asked whether the earlier 30-epoch row-split gated MLP run was too short.
- Completed a GPU confirmation run with the gated four-feature MLP:
  - Result root: `/hy-tmp/wan22_adaptive_threshold_mlp_gated_4feature_rowsplit_gpu_long100_20260630_015638`
  - Workspace symlink: `experiment_results/wan22_adaptive_threshold_mlp_gated_4feature_rowsplit_gpu_long100_20260630_015638`
  - Feature cache: `/hy-tmp/wan22_adaptive_threshold_feature_cache_candidate_inverse_20260616_012409`
  - Features: `latent_pool`, `temporal_var`, `frame_diff_mean`, `frame_diff_var`
  - Split mode: `row`
  - Training: `100` epochs, `early_stop_patience=20`, `min_lr=1e-6`, batch size `256`, CUDA.
- Result:
  - Completed all 100 epochs; no early stop.
  - Parameter count: `71,045`.
  - Best epoch: `97`.
  - Best validation MAE: `0.06126960859298706`.
  - Best validation loss: `0.052473511600494384`.
  - Train MAE at best epoch: `0.0634434845218435`.
  - Final epoch validation MAE: `0.06131729580387473`.
- Comparison:
  - Earlier gated MLP row split 30-epoch run best validation MAE: `0.07593761396706104`.
  - In this long run, epoch 30 validation MAE was `0.07319095213487745`, and it kept improving to `0.06126960859298706`.
  - Relative improvement from the earlier 30-epoch result to the 100-epoch best is about `19.3%`; relative improvement from this run's epoch 30 to epoch 97 is about `16.3%`.
  - Conclusion: 30 epochs were materially too short for this gated MLP row-split setting, but training longer still does not close the gap to the row-split MiniDiT reference best validation MAE `0.03800193872973323`.
- Best-epoch diagnostics:
  - Gate means from saved validation predictions: `latent_pool=0.4814`, `temporal_var=0.1602`, `frame_diff_mean=0.1454`, `frame_diff_var=0.2130`.
  - Validation MAE by threshold: `0.10=0.0132`, `0.15=0.0289`, `0.20=0.0372`, `0.25=0.0523`, `0.30=0.0575`, `0.40=0.0848`, `0.50=0.0724`, `0.60=0.0695`, `0.70=0.0657`, `0.80=0.1338`.
  - Validation MAE by step range: `step_00_09=0.0821`, `step_10_39=0.0565`, `step_40_49=0.0545`.
- No commit was made.

## 2026-06-30 Gated MLP 5-Feature Retrain

- Corrected the gated multi-feature MLP default from 4 features to 5 features by adding the omitted `temporal_mean` feature.
- Current recommended gated feature set:
  - `latent_pool`
  - `temporal_mean`
  - `temporal_var`
  - `frame_diff_mean`
  - `frame_diff_var`
- Updated:
  - `adaptive_threshold_predictor/models.py`
  - `adaptive_threshold_predictor/README.md`
  - `reports/report_gated_multifeature_mlp_architecture.md`
- Verified parameter count for the 5-feature gated MLP:
  - `83,526` trainable parameters
  - feature encoders `62,080`
  - condition encoder `4,352`
  - gate head `4,485`
  - prediction head `12,609`
- Removed superseded 4-feature training outputs and workspace symlinks:
  - `/hy-tmp/wan22_adaptive_threshold_mlp_gated_4feature_samplesplit_20260630_013006`
  - `/hy-tmp/wan22_adaptive_threshold_mlp_gated_4feature_rowsplit_gpu_20260630_014852`
  - `/hy-tmp/wan22_adaptive_threshold_mlp_gated_4feature_rowsplit_gpu_long100_20260630_015638`
  - matching `experiment_results/wan22_adaptive_threshold_mlp_gated_4feature_*` symlinks
- New 5-feature result roots:
  - `/hy-tmp/wan22_adaptive_threshold_mlp_gated_5feature_samplesplit_20260630_021641`
  - `/hy-tmp/wan22_adaptive_threshold_mlp_gated_5feature_rowsplit_gpu_20260630_021641`
  - `/hy-tmp/wan22_adaptive_threshold_mlp_gated_5feature_rowsplit_gpu_long100_20260630_021641`
  - matching symlinks exist under `experiment_results/`
- Training commands are archived in each result root under `commands/launch_train.sh`; raw logs are under `logs/train.log`.
- 5-feature training results:

| Run | Split | Epochs Run | Best Epoch | Best Val MAE | Best Val Loss | Final Val MAE |
|---|---|---:|---:|---:|---:|---:|
| gated 5-feature sample split 30 | sample | 12 / 30, early stopped | 7 | `0.1142528785` | `0.1050056725` | `0.1178081666` |
| gated 5-feature row split 30 | row | 30 / 30 | 30 | `0.0756697811` | `0.0667008773` | `0.0756697811` |
| gated 5-feature row split 100 | row | 100 / 100 | 98 | `0.0601118673` | `0.0513865515` | `0.0601155481` |

- Mean validation gate weights:

| Run | latent_pool | temporal_mean | temporal_var | frame_diff_mean | frame_diff_var |
|---|---:|---:|---:|---:|---:|
| sample split 30 | `0.5146` | `0.2312` | `0.0784` | `0.0689` | `0.1068` |
| row split 30 | `0.4112` | `0.2113` | `0.1137` | `0.1136` | `0.1502` |
| row split 100 | `0.3906` | `0.1825` | `0.1439` | `0.1116` | `0.1714` |

- Comparison against superseded 4-feature gated MLP:
  - sample split 30 became worse: old `0.1115179658` -> new `0.1142528785`.
  - row split 30 improved marginally: old `0.0759376140` -> new `0.0756697811`.
  - row split 100 improved modestly: old `0.0612696086` -> new `0.0601118673`.
- Interpretation:
  - Adding `temporal_mean` helps row-split validation slightly, especially in the 100-epoch confirmation run.
  - The sample-split result does not improve and early-stops at epoch 12.
  - Even the 5-feature row-split 100 result remains substantially behind the current MiniDiT row-split reference MAE `0.0380019387`.
- Validation:
  - `py_compile` passed for `adaptive_threshold_predictor/models.py`, `adaptive_threshold_predictor/data.py`, and `adaptive_threshold_predictor/train_gate.py`.
  - `git diff --check` passed.
  - No `adaptive_threshold_predictor.train_gate` processes remained after training.
  - GPU was idle after completion: `0 MiB / 81920 MiB`, no running GPU processes.
- Session log: `logs/session_20260630_gated_mlp_5feature_retrain.md`.
- No commit was made.

## 2026-06-30 Gated MLP Range-Constrained Output Retrain

- User identified a semantic mismatch: the 5-feature gated MLP output head used
  a direct sigmoid output range `[0, 1]`, while the MiniDiT/Transformer predictor
  used a scaled threshold range.
- Updated the MLP-family threshold heads to use:

```text
threshold = min_threshold + sigmoid(raw) * (max_threshold - min_threshold)
```

- Current defaults are `min_threshold=0.10` and `max_threshold=0.80`.
- Updated `adaptive_threshold_predictor/models.py` so relevant MLP/condition
  classes emit raw logits and then apply the scaled sigmoid mapping:
  - `ImprovedAdaCacheGate`
  - `CachedFeatureAdaCacheGate`
  - `GatedFeatureFusionAdaCacheGate`
  - `CachedGatedFeatureAdaCacheGate`
  - `GatedMultiFeatureAdaCacheGate`
  - `ConditionOnlyAdaCacheGate`
- Updated `adaptive_threshold_predictor/train_gate.py` so `build_model()` passes
  `--min_threshold` and `--max_threshold` into the MLP-family model constructors.
- Updated documentation:
  - `adaptive_threshold_predictor/README.md`
  - `reports/report_gated_multifeature_mlp_architecture.md`
- Parameter count is unchanged:
  - 5-feature gated MLP: `83,526` trainable parameters.
- Validation before full retraining:
  - `py_compile` passed for `adaptive_threshold_predictor/models.py`,
    `adaptive_threshold_predictor/train_gate.py`, and
    `adaptive_threshold_predictor/data.py`.
  - Random forward smoke for `CachedGatedFeatureAdaCacheGate` with five features
    produced `[B, 1]` outputs inside `[0.10, 0.80]`.
- Removed temporary CPU smoke output:
  - `/hy-tmp/wan22_adaptive_threshold_mlp_gated_5feature_range_smoke_20260630_0345`
- New range-constrained result roots:
  - `/hy-tmp/wan22_adaptive_threshold_mlp_gated_5feature_range_samplesplit_20260630_035000`
  - `/hy-tmp/wan22_adaptive_threshold_mlp_gated_5feature_range_rowsplit_gpu_20260630_035000`
  - `/hy-tmp/wan22_adaptive_threshold_mlp_gated_5feature_range_rowsplit_gpu_long100_20260630_035000`
  - matching symlinks exist under `experiment_results/`
- Training commands are archived in each result root under `commands/launch_train.sh`; logs are under `logs/train.log`.
- Range-constrained 5-feature results:

| Run | Split | Epochs Run | Best Epoch | Best Val MAE | Best Val Loss | Final Val MAE | Best Prediction Range |
|---|---|---:|---:|---:|---:|---:|---|
| range sample split 30 | sample | 14 / 30, early stopped | 9 | `0.1143567288` | `0.1049280047` | `0.1278297383` | `[0.1023, 0.7303]` |
| range row split 30 | row | 30 / 30 | 30 | `0.0770497653` | `0.0678418150` | `0.0770497653` | `[0.1040, 0.7743]` |
| range row split 100 | row | 100 / 100 | 98 | `0.0610311001` | `0.0522424302` | `0.0610505000` | `[0.1010, 0.7985]` |

- Mean validation gate weights:

| Run | latent_pool | temporal_mean | temporal_var | frame_diff_mean | frame_diff_var |
|---|---:|---:|---:|---:|---:|
| sample split 30 | `0.5464` | `0.1839` | `0.0945` | `0.0986` | `0.0766` |
| row split 30 | `0.4781` | `0.1967` | `0.1267` | `0.0998` | `0.0987` |
| row split 100 | `0.3987` | `0.2533` | `0.0942` | `0.0962` | `0.1576` |

- Comparison against the previous direct-sigmoid `[0, 1]` 5-feature runs:

| Split / Budget | Previous Best Val MAE | Range-Constrained Best Val MAE | Delta |
|---|---:|---:|---:|
| sample split 30 | `0.1142528785` | `0.1143567288` | `+0.0001038503` |
| row split 30 | `0.0756697811` | `0.0770497653` | `+0.0013799842` |
| row split 100 | `0.0601118673` | `0.0610311001` | `+0.0009192327` |

- Interpretation:
  - The output mismatch is fixed; saved predictions now stay inside the intended
    `[0.10, 0.80]` threshold range.
  - Offline threshold MAE did not improve: sample split is essentially
    unchanged, and row split is slightly worse.
  - Row split still benefits from longer training (`0.07705` at 30 epochs to
    `0.06103` at 100 epochs), but the gated MLP remains behind the MiniDiT
    row-split reference MAE `0.0380019387`.
- Final checks:
  - `py_compile` passed.
  - `git diff --check` passed.
  - No `adaptive_threshold_predictor.train_gate` process remained after
    training.
- Session log: `logs/session_20260630_gated_mlp_range_retrain.md`.
- No commit was made.

## 2026-06-30 SeaCache Sampler Comparison Availability Check

- User asked whether complete SeaCache `dpm++` and `unipc` performance comparison results exist.
- Checked current result roots and summary/aggregate tables.
- Complete full VBench10 comparison exists:
  - `dpm++`: `/hy-tmp/wan22_vbench10_three_cache_full_merge_and_timestep_only_full_20260623/timestep_only_seacache_vbench10_full/wan22_seacache_vbench10_50step_45f_480p_20260618_161845/merged/summary.csv`
  - `dpm++` aggregate: `/hy-tmp/wan22_vbench10_three_cache_full_merge_and_timestep_only_full_20260623/timestep_only_seacache_vbench10_full/wan22_seacache_vbench10_50step_45f_480p_20260618_161845/merged/aggregate_by_threshold.csv`
  - `unipc`: `/hy-tmp/wan22_seacache_unipc_vbench10_50step_45f_480p_20260627_023222/results/summary.csv`
  - `unipc` aggregate: `/hy-tmp/wan22_seacache_unipc_vbench10_50step_45f_480p_20260627_023222/results/aggregate_by_threshold.csv`
  - Both have 10 VBench10 samples; overlapping thresholds are `0.10`, `0.20`, `0.30`, and `0.50`.
- Complete full Ali-10 comparison does not exist:
  - full `unipc` Ali-10 exists at `/hy-tmp/wan22_seacache_unipc_ali10_50step_45f_480p_20260627_023222/results/summary.csv`.
  - full `dpm++` Ali-10 remains missing; only Ali prompt 1/2 pilot results exist under `/hy-tmp/wan22_seacache_50step_45f_480p_20260611_191733`, `/hy-tmp/wan22_seacache_prompt02_dense_20260611_204826`, and `/hy-tmp/wan22_seacache_prompt02_highthr_20260612_000218`.
- Therefore: full sampler comparison is available for VBench10, but not for Ali-10.

## 2026-06-30 Sampling Solver Impact Report Update

- User asked to add the VBench10 SeaCache sampler comparison, mimic the existing format, rename the report to sampling-solver impact, and split it into SeaCache and ZEUS parts.
- Replaced old ZEUS-only solver comparison report path with the new report:
  - old: `reports/report_zeus_solver_ali10_vbench10_comparison_20260624.md`
  - new: `reports/report_sampling_solver_impact_zeus_seacache_20260630.md`
- New report title: `Sampling Solver Impact on ZEUS and SeaCache`.
- Preserved the ZEUS aggregate and per-sample comparisons for ali-10 and VBench10.
- Added SeaCache VBench10:
  - source artifact table for `dpm++` and `unipc` summary/aggregate CSVs.
  - aggregate comparison for thresholds `0.10`, `0.20`, `0.30`, and `0.50`.
  - per-sample comparison tables for each overlapping threshold.
- Kept the report in a table-focused format and removed explanatory/takeaway text per user request.
- Added a data-coverage table clarifying that full SeaCache sampler comparison exists for VBench10 but not for Ali-10, because full `SeaCache + dpm++ + ali-10` remains missing.
- Validation:
  - verified SeaCache source CSVs exist and row counts match: `dpm++` aggregate 10 thresholds, `unipc` aggregate 4 thresholds, both summaries have 10 VBench10 samples and 10 rows for each overlapping threshold.
  - manually recomputed overlapping SeaCache aggregate deltas from source CSVs and confirmed the report values.
  - `git diff --check -- reports/report_sampling_solver_impact_zeus_seacache_20260630.md reports/report_zeus_solver_ali10_vbench10_comparison_20260624.md` passed.

## 2026-06-30 Wan2.1 vs Wan2.2 Model Difference Report Merge

- User asked to simplify two externally written reports:
  - `reports/report_seacache_wan21_wan22_ali10_unipc_2026-06-30.md`
  - `reports/report_zeus_wan21_wan22_ali10_unipc_2026-06-30.md`
- Removed the sampler-difference comparison section from the SeaCache report, including the first-two-prompt DPM++ vs UniPC comparison and remaining sampler-mismatch wording.
- Added a unified table-focused model-difference report:
  - `reports/report_model_difference_zeus_seacache_wan21_wan22_20260630.md`
- The new report follows the compact format of `reports/report_sampling_solver_impact_zeus_seacache_20260630.md`:
  - data coverage.
  - shared configuration.
  - source artifacts.
  - ZEUS method/config/results.
  - SeaCache method/config/results.
  - combined model-difference summary and caveats.
- No experiments were run and no commit was made.
- Session log: `logs/session_20260630_model_difference_report_merge.md`.

Follow-up edit in the same report:

- User requested removing all explanatory text and keeping only each experiment's configuration, complete results, and aggregate results.
- Rewrote `reports/report_model_difference_zeus_seacache_wan21_wan22_20260630.md` as a table-only report:
  - ZEUS experiment configuration.
  - ZEUS schedule configuration.
  - ZEUS aggregate results.
  - per-sample results for Wan2.1 official demo, Wan2.1 strict Euler, Wan2.1 strict UniPC, and Wan2.2 strict UniPC high/low reset.
  - Wan2.1 strict UniPC vs Wan2.2 strict UniPC aggregate and per-sample comparisons.
  - SeaCache experiment configuration.
  - Wan2.2 SeaCache cache configuration.
  - SeaCache aggregate results.
  - per-sample results for Wan2.1 Ali-10 and Wan2.2 Ali-10.
  - Wan2.1 vs Wan2.2 SeaCache aggregate and per-sample comparisons.
- Removed method introductions, logic descriptions, interpretation, caveats, and narrative conclusion sections.

## 2026-06-30 5-Feature Gated MLP Online Adaptive SeaCache Queue

- User noted that a Transformer/MiniDiT adaptive predictor online validation was
  currently running under `adaptive_seacache_wan22` and requested the same
  validation for the 5-feature gated MLP architecture.
- Current running Transformer validation:
  - tmux session: `wan22_adaptive_mini_dit_split_20260630_025328`
  - result root: `/hy-tmp/wan22_adaptive_seacache_mini_dit_split_compare_50step_45f_480p_20260630_025328`
  - runner: `experiments/adaptive_seacache_mini_dit_split_compare_50step_45f_480p/run_batch.py`
  - protocol: `24` candidates = `2` splits (`sample_split`, `row_split`) * `2`
    target PSNRs (`22`, `28`) * `2` datasets (`VBench10`, `OpenVid100 train`)
    * `3` prompts.
  - It reuses existing no-cache baselines and loads the WanT2V pipeline once.
- Added online inference support for the 5-feature gated MLP checkpoint:
  - `adaptive_seacache_wan22/cache.py`
    - added `mlp_gated` auto-detection for checkpoints with `fusion.*` state keys.
    - added online five-feature extraction:
      `latent_pool`, `temporal_mean`, `temporal_var`,
      `frame_diff_mean`, `frame_diff_var`.
    - loads `feature_sets`, `hidden_dim`, `feature_embedding_dim`,
      `psnr_min/max`, and `min/max_threshold` from checkpoint/config metadata.
  - `adaptive_seacache_wan22/generate_t2v.py`
    - added `mlp_gated` to `--adaptive_model_type` choices.
- Validation:
  - `py_compile` passed for `adaptive_seacache_wan22/cache.py`,
    `adaptive_seacache_wan22/generate_t2v.py`, and
    `adaptive_threshold_predictor/models.py`.
  - Loaded
    `/hy-tmp/wan22_adaptive_threshold_mlp_gated_5feature_range_samplesplit_20260630_035000/best_model_checkpoint.pt`
    with `model_type=auto`; it resolved to `mlp_gated`, feature sets matched
    the five-feature list, hidden dim and feature embedding dim were both `64`,
    range was `[0.1, 0.8]`, and a random-latent forward prediction succeeded
    inside that range.
  - CPU validation of the online runner with 5-feature checkpoints found the
    expected `24` candidates and reusable baselines.
- Added launch helper:
  - `experiments/adaptive_seacache_mlp_gated_5feature_split_compare_50step_45f_480p/run_tmux.sh`
- Queued the 5-feature online validation behind the currently running
  Transformer validation to avoid competing for the single A100:
  - tmux session: `wan22_adaptive_mlp_gated5_split_20260630_050727`
  - result root:
    `/hy-tmp/wan22_adaptive_seacache_mlp_gated_5feature_range_split_compare_50step_45f_480p_20260630_050727`
  - symlink:
    `experiment_results/wan22_adaptive_seacache_mlp_gated_5feature_range_split_compare_50step_45f_480p_20260630_050727`
  - queued command waits for tmux session
    `wan22_adaptive_mini_dit_split_20260630_025328` to exit, then runs the same
    24-candidate protocol with:
    - sample split checkpoint:
      `/hy-tmp/wan22_adaptive_threshold_mlp_gated_5feature_range_samplesplit_20260630_035000/best_model_checkpoint.pt`
    - row split checkpoint:
      `/hy-tmp/wan22_adaptive_threshold_mlp_gated_5feature_range_rowsplit_gpu_long100_20260630_035000/best_model_checkpoint.pt`
    - sample split JSON:
      `/hy-tmp/wan22_adaptive_threshold_mlp_gated_5feature_range_samplesplit_20260630_035000/split.json`
    - `--adaptive_min_threshold 0.10`
    - `--adaptive_max_threshold 0.80`
- At queue launch time, the Transformer run had `22` completed summary rows and
  was running `openvid100_train_openvid_005_row_split_target_22`, candidate
  `23/24`.
- Monitor:
  - Transformer log:
    `/hy-tmp/wan22_adaptive_seacache_mini_dit_split_compare_50step_45f_480p_20260630_025328/logs/runner.log`
  - 5-feature log after it starts:
    `/hy-tmp/wan22_adaptive_seacache_mlp_gated_5feature_range_split_compare_50step_45f_480p_20260630_050727/logs/runner.log`
- Session log: `logs/session_20260630_gated_mlp_online_validation_queue.md`.
- No commit was made.

Completion update:

- The queued 5-feature gated MLP online validation completed successfully.
- Final status:
  - tmux session exited.
  - `summary.csv` has `24` completed candidate rows.
  - `failed/` is empty.
  - runner log ends with:
    `Completed experiment: /hy-tmp/wan22_adaptive_seacache_mlp_gated_5feature_range_split_compare_50step_45f_480p_20260630_050727`
- Result files:
  - summary:
    `/hy-tmp/wan22_adaptive_seacache_mlp_gated_5feature_range_split_compare_50step_45f_480p_20260630_050727/results/summary.csv`
  - aggregate:
    `/hy-tmp/wan22_adaptive_seacache_mlp_gated_5feature_range_split_compare_50step_45f_480p_20260630_050727/results/aggregate_by_dataset_model_target.csv`
- 5-feature aggregate results:

| Dataset | Split | Target | Completed | Overall Speedup | Mean PSNR | Target Error | Mean Reuse Decisions | Mean Threshold |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| OpenVid train | row | 22 | 3 | `2.523x` | `21.970` | `-0.030` | `64.7` | `0.488` |
| OpenVid train | row | 28 | 3 | `1.773x` | `26.095` | `-1.905` | `46.0` | `0.279` |
| OpenVid train | sample | 22 | 3 | `2.559x` | `22.858` | `+0.858` | `65.3` | `0.495` |
| OpenVid train | sample | 28 | 3 | `1.718x` | `27.365` | `-0.635` | `44.0` | `0.291` |
| VBench10 | row | 22 | 3 | `2.065x` | `20.810` | `-1.190` | `54.7` | `0.353` |
| VBench10 | row | 28 | 3 | `1.627x` | `24.484` | `-3.516` | `40.0` | `0.201` |
| VBench10 | sample | 22 | 3 | `2.229x` | `17.159` | `-4.841` | `58.7` | `0.358` |
| VBench10 | sample | 28 | 3 | `1.539x` | `25.460` | `-2.540` | `36.0` | `0.184` |

- Same-protocol comparison against MiniDiT aggregate:

| Dataset | Split | Target | 5-feature Speedup | MiniDiT Speedup | Speed Delta | 5-feature PSNR | MiniDiT PSNR | PSNR Delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| OpenVid train | row | 22 | `2.523x` | `2.447x` | `+0.077x` | `21.970` | `23.007` | `-1.037` |
| OpenVid train | row | 28 | `1.773x` | `1.633x` | `+0.140x` | `26.095` | `27.710` | `-1.614` |
| OpenVid train | sample | 22 | `2.559x` | `2.598x` | `-0.039x` | `22.858` | `22.151` | `+0.706` |
| OpenVid train | sample | 28 | `1.718x` | `1.794x` | `-0.076x` | `27.365` | `29.019` | `-1.654` |
| VBench10 | row | 22 | `2.065x` | `2.068x` | `-0.003x` | `20.810` | `20.466` | `+0.344` |
| VBench10 | row | 28 | `1.627x` | `1.539x` | `+0.088x` | `24.484` | `25.469` | `-0.984` |
| VBench10 | sample | 22 | `2.229x` | `2.113x` | `+0.116x` | `17.159` | `16.737` | `+0.422` |
| VBench10 | sample | 28 | `1.539x` | `1.582x` | `-0.043x` | `25.460` | `23.796` | `+1.664` |

- Immediate read:
  - The 5-feature gated MLP is not uniformly worse online despite weaker
    offline MAE.
  - On OpenVid target 28, MiniDiT has clearly higher PSNR.
  - On VBench10 sample split target 28, 5-feature MLP has higher PSNR than
    MiniDiT in this 3-prompt pilot.
  - Both predictors miss VBench target 22 badly under sample split, indicating
    the online target-control problem remains unresolved.
- Note: after completion, `nvidia-smi` returned `No devices were found`, so the
  GPU appears to have been disabled or detached after the run. This was not
  changed by the agent.

## 2026-06-30 MiniDiT Split Compare Progress Check

- Checked the running MiniDiT row-split vs sample-split online validation requested for adaptive SeaCache.
- Current tmux sessions:
  - `wan22_adaptive_mini_dit_split_20260630_025328` is still running.
  - `wan22_adaptive_mlp_gated5_split_20260630_050727` is queued and waiting for the MiniDiT session to exit before starting.
- MiniDiT result root:
  `/hy-tmp/wan22_adaptive_seacache_mini_dit_split_compare_50step_45f_480p_20260630_025328`
- Status at 2026-06-30 05:15 CST:
  - `results/summary.csv` has `23/24` completed rows.
  - `failed/` is empty.
  - GPU is active on `NVIDIA A100 80GB PCIe`, about `47317/81920 MiB` used, `100%` utilization.
  - Last running candidate:
    `openvid100_train_openvid_005_row_split_target_28`.
  - tmux pane showed this last candidate at `17/50` sampling steps.
- Current aggregate at 23 rows:
  - `vbench10/sample_split/target22`: speedup `2.113x`, mean PSNR `16.737`, target error `-5.263`.
  - `vbench10/sample_split/target28`: speedup `1.582x`, mean PSNR `23.796`, target error `-4.204`.
  - `vbench10/row_split/target22`: speedup `2.068x`, mean PSNR `20.466`, target error `-1.534`.
  - `vbench10/row_split/target28`: speedup `1.539x`, mean PSNR `25.469`, target error `-2.531`.
  - `openvid100_train/sample_split/target22`: speedup `2.598x`, mean PSNR `22.151`, target error `+0.151`.
  - `openvid100_train/sample_split/target28`: speedup `1.794x`, mean PSNR `29.019`, target error `+1.019`.
  - `openvid100_train/row_split/target22`: speedup `2.447x`, mean PSNR `23.007`, target error `+1.007`.
  - `openvid100_train/row_split/target28`: currently `2/3` rows complete, speedup `1.801x`, mean PSNR `28.556`, target error `+0.556`.
- No code changes were made during this check.
- Session log: `logs/session_20260630_mini_dit_split_compare_progress_check.md`.

## 2026-06-30 MiniDiT Split Compare Completed

- Rechecked MiniDiT row-split vs sample-split online validation.
- Completed `24/24` candidates with `0` failures.
- Result root:
  `/hy-tmp/wan22_adaptive_seacache_mini_dit_split_compare_50step_45f_480p_20260630_025328`
- Final result tables:
  - summary:
    `/hy-tmp/wan22_adaptive_seacache_mini_dit_split_compare_50step_45f_480p_20260630_025328/results/summary.csv`
  - aggregate:
    `/hy-tmp/wan22_adaptive_seacache_mini_dit_split_compare_50step_45f_480p_20260630_025328/results/aggregate_by_dataset_model_target.csv`
- Final aggregate:
  - `vbench10/sample_split/target22`: speedup `2.113x`, mean PSNR `16.737`, target error `-5.263`, mean threshold `0.359`.
  - `vbench10/sample_split/target28`: speedup `1.582x`, mean PSNR `23.796`, target error `-4.204`, mean threshold `0.211`.
  - `vbench10/row_split/target22`: speedup `2.068x`, mean PSNR `20.466`, target error `-1.534`, mean threshold `0.329`.
  - `vbench10/row_split/target28`: speedup `1.539x`, mean PSNR `25.469`, target error `-2.531`, mean threshold `0.178`.
  - `openvid100_train/sample_split/target22`: speedup `2.598x`, mean PSNR `22.151`, target error `+0.151`, mean threshold `0.487`.
  - `openvid100_train/sample_split/target28`: speedup `1.794x`, mean PSNR `29.019`, target error `+1.019`, mean threshold `0.252`.
  - `openvid100_train/row_split/target22`: speedup `2.447x`, mean PSNR `23.007`, target error `+1.007`, mean threshold `0.484`.
  - `openvid100_train/row_split/target28`: speedup `1.633x`, mean PSNR `27.710`, target error `-0.290`, mean threshold `0.232`.
- Split comparison:
  - On VBench10, row split is closer to target than sample split for both targets:
    absolute target error improves by `3.729 dB` at target 22 and `1.673 dB` at target 28, with about `0.04x` lower speedup.
  - On OpenVid train, sample split is closer at target 22, while row split is closer at target 28. Row split is slower by about `0.15x-0.16x`.
  - Both MiniDiT splits still undershoot VBench10 target PSNR; row split undershoots less.
- The queued 5-feature gated MLP validation started after MiniDiT completed:
  - tmux session: `wan22_adaptive_mlp_gated5_split_20260630_050727`
  - result root:
    `/hy-tmp/wan22_adaptive_seacache_mlp_gated_5feature_range_split_compare_50step_45f_480p_20260630_050727`
  - current check showed `1/24` completed, `0` failures, and it was running `vbench10_vbench10_001_sample_split_target_28`.
- No code changes were made during this check.
- Session log: `logs/session_20260630_mini_dit_split_compare_completed.md`.

## 2026-06-30 MiniDiT Row-Split vs Fixed SeaCache VBench10 Comparison

- Compared MiniDiT adaptive SeaCache row-split results on the tested VBench10 prompts
  (`vbench10_001`, `vbench10_002`, `vbench10_003`) against normal fixed-threshold
  SeaCache dpm++ results for the same prompts.
- Sources:
  - MiniDiT row-split summary:
    `/hy-tmp/wan22_adaptive_seacache_mini_dit_split_compare_50step_45f_480p_20260630_025328/results/summary.csv`
  - fixed SeaCache summary:
    `/hy-tmp/wan22_vbench10_three_cache_full_merge_and_timestep_only_full_20260623/timestep_only_seacache_vbench10_full/wan22_seacache_vbench10_50step_45f_480p_20260618_161845/merged/summary.csv`
- Three-prompt fixed SeaCache aggregate:
  - threshold `0.10`: speedup `1.109x`, mean PSNR `35.623`
  - threshold `0.15`: speedup `1.410x`, mean PSNR `27.728`
  - threshold `0.20`: speedup `1.575x`, mean PSNR `24.272`
  - threshold `0.25`: speedup `1.844x`, mean PSNR `23.614`
  - threshold `0.30`: speedup `1.979x`, mean PSNR `22.218`
  - threshold `0.40`: speedup `2.425x`, mean PSNR `17.364`
  - threshold `0.50`: speedup `2.753x`, mean PSNR `17.350`
  - threshold `0.60`: speedup `3.125x`, mean PSNR `16.618`
  - threshold `0.70`: speedup `3.386x`, mean PSNR `16.756`
  - threshold `0.80`: speedup `3.534x`, mean PSNR `16.539`
- Aggregate target comparison on the same three prompts:
  - target 22:
    - MiniDiT row split: speedup `2.068x`, mean PSNR `20.466`, target error `-1.534`
    - best fixed SeaCache by mean PSNR closeness: threshold `0.30`, speedup `1.979x`, mean PSNR `22.218`, target error `+0.218`
    - MiniDiT is `+0.089x` faster but `1.752 dB` lower in mean PSNR.
  - target 28:
    - MiniDiT row split: speedup `1.539x`, mean PSNR `25.469`, target error `-2.531`
    - best fixed SeaCache by mean PSNR closeness: threshold `0.15`, speedup `1.410x`, mean PSNR `27.728`, target error `-0.272`
    - MiniDiT is `+0.129x` faster but `2.259 dB` lower in mean PSNR.
- Takeaway: on these VBench10 prompts, MiniDiT row split does not beat fixed
  SeaCache when selecting a fixed threshold by target PSNR closeness. It is a bit
  faster than the closest fixed-threshold points, but its target PSNR undershoot
  is materially worse.
- Session log: `logs/session_20260630_mini_dit_vs_fixed_seacache_vbench10.md`.

## 2026-06-30 MiniDiT Transformer Predictor Comprehensive Report

- Wrote a comprehensive report for the Transformer-style adaptive threshold predictor:
  `reports/report_mini_dit_transformer_predictor_comprehensive_20260630.md`.
- The report includes:
  - existing architecture diagram:
    `reports/assets/mini_dit_cls_predictor_architecture.svg`
  - architecture parameter settings and module parameter counts.
  - sample-split and row-split training settings.
  - offline training metrics and train/validation loss curves.
  - online inference protocol and per-prompt plus aggregate results for the
    24-candidate VBench10/OpenVid train run.
- Added generated loss curve asset:
  `reports/assets/mini_dit_cls_training_loss_curves.svg`.
- Notes captured in the report:
  - row split has much lower offline validation MAE but is not a strict sample-level generalization test.
  - online VBench10 row split is better calibrated than sample split, but still undershoots target PSNR.
  - online predictor timing fields were not populated in this runner, so online performance is reported with compute elapsed/speedup rather than predictor overhead.
- Validation:
  - `git diff --check` passed for the new report, loss curve asset, progress update, and session log.
- Session log: `logs/session_20260630_mini_dit_transformer_predictor_report.md`.

## 2026-06-30 5-Feature Gated MLP Predictor Comprehensive Report

- User requested a comprehensive 5-feature gated MLP predictor report modeled
  after `reports/report_mini_dit_transformer_predictor_comprehensive_20260630.md`.
- Wrote:
  - `reports/report_gated_multifeature_mlp_predictor_comprehensive_20260630.md`
- Added generated loss-curve asset:
  - `reports/assets/gated_multifeature_mlp_training_loss_curves.svg`
- The report includes:
  - existing architecture diagram:
    `reports/assets/gated_multifeature_mlp_architecture.svg`
  - architecture flow, module settings, feature definitions, and parameter
    count (`83,526` trainable parameters).
  - training parameters for:
    - sample split 30-epoch budget,
    - row split 30-epoch budget,
    - row split 100-epoch budget.
  - offline training metrics, last-epoch metrics, and mean gate weights.
  - train/test Smooth L1 loss curves, with train/test overlaid per run and
    separate subplots for sample 30, row 30, and row 100.
  - explicit row-split 30-epoch vs 100-epoch comparison:
    - row split 30 run best test MAE: `0.077050`
    - row split 100 epoch-30 test MAE: `0.074371`
    - row split 100 best test MAE: `0.061031` at epoch `98`
    - 100-epoch best improves over independent row-30 best by about `20.8%`.
  - online inference settings and real T2V results from:
    `/hy-tmp/wan22_adaptive_seacache_mlp_gated_5feature_range_split_compare_50step_45f_480p_20260630_050727`
  - per-prompt and aggregate online results for VBench10 and OpenVid100 train.
  - same-protocol comparison against the MiniDiT online result aggregate.
- Main report takeaways:
  - 5-feature gated MLP is much smaller than MiniDiT (`83,526` vs `724,513`
    params).
  - row split offline metrics are much better than sample split but are not a
    held-out-sample generalization test.
  - longer row-split training matters; sample split early-stops instead.
  - online results are mixed rather than uniformly worse than MiniDiT.
  - target control remains weak on VBench10, supporting the earlier conclusion
    that the `candidate_inverse` training target and online adaptive control are
    still mismatched.
- Validation:
  - generated report and SVG exist.
  - `git diff --check` passed for the report and loss curve asset before the
    progress/session-log update.
- Session log: `logs/session_20260630_gated_mlp_predictor_comprehensive_report.md`.
- No commit was made.

Follow-up comparison-report split:

- User requested moving `9. Same-Protocol MiniDiT Comparison` out of the
  5-feature comprehensive report into a dedicated comparison report.
- Updated:
  - `reports/report_gated_multifeature_mlp_predictor_comprehensive_20260630.md`
    - removed the old same-protocol MiniDiT comparison section.
    - renumbered `Findings` and `Artifacts`.
    - kept the 5-feature report focused on the 5-feature architecture, training,
      loss curves, and online results.
    - artifact table now points to the dedicated comparison report.
- Added:
  - `reports/report_adaptive_predictor_mini_dit_vs_gated_mlp_comparison_20260630.md`
- Dedicated comparison report includes:
  - method summary for MiniDiT and 5-feature gated MLP.
  - training loss data table for both methods:
    - MiniDiT sample split 30.
    - MiniDiT row split 30.
    - 5-feature MLP sample split 30.
    - 5-feature MLP row split 30.
    - 5-feature MLP row split 100.
  - online-checkpoint training comparison table.
  - training loss figures for both methods:
    - `reports/assets/mini_dit_cls_training_loss_curves.svg`
    - `reports/assets/gated_multifeature_mlp_training_loss_curves.svg`
  - same online inference protocol table.
  - aggregate online comparison table with speed/PSNR/target-error/threshold
    deltas.
  - per-prompt online comparison table.
  - findings and artifact paths.
- No new experiments were run.
- Session log:
  `logs/session_20260630_predictor_comparison_report_split.md`.
- No commit was made.

## 2026-06-30 Transformer vs 5-Feature Predictor Result Readout

- Read and cross-checked:
  - `reports/report_mini_dit_transformer_predictor_comprehensive_20260630.md`
  - `reports/report_gated_multifeature_mlp_predictor_comprehensive_20260630.md`
  - `reports/report_adaptive_predictor_mini_dit_vs_gated_mlp_comparison_20260630.md`
  - MiniDiT online summary:
    `/hy-tmp/wan22_adaptive_seacache_mini_dit_split_compare_50step_45f_480p_20260630_025328/results/summary.csv`
  - 5-feature online summary:
    `/hy-tmp/wan22_adaptive_seacache_mlp_gated_5feature_range_split_compare_50step_45f_480p_20260630_050727/results/summary.csv`
- Recomputed lightweight online aggregates with Python standard-library CSV
  parsing.
- Key readout:
  - MiniDiT is clearly stronger offline on row split (`0.038002` MAE vs
    5-feature row-100 `0.061031` MAE), while sample split is essentially tied
    (`0.114459` vs `0.114357` MAE).
  - Online, neither model has reliable target-quality control. Across the 24
    candidates, both hit the requested PSNR target in only `8/24` cases.
  - MiniDiT online average: speedup `2.010x`, PSNR `23.544 dB`, mean target
    error `-1.456 dB`, mean absolute target error `3.148 dB`.
  - 5-feature online average: speedup `2.065x`, PSNR `23.275 dB`, mean target
    error `-1.725 dB`, mean absolute target error `2.813 dB`.
  - MiniDiT tends to preserve higher PSNR on OpenVid target-28 cells; 5-feature
    MLP is slightly faster on average and has lower aggregate absolute target
    error, but no consistent dominance.
  - VBench10 target control remains poor for both, especially target 22 under
    sample split; this reinforces that the `candidate_inverse` offline task is
    not well aligned with online adaptive control.
- No code or experiment runner changes were made.
- Session log:
  `logs/session_20260630_transformer_vs_5feature_readout.md`.
- No commit was made.

## 2026-06-30 Row Split vs Sample Split Readout

- Compared row split and sample split results for both current adaptive
  threshold predictors.
- Main readout:
  - Row split is much easier offline because train/test share source-video
    identities; it measures same-video interpolation rather than held-out-video
    generalization.
  - MiniDiT row split improves dramatically over sample split (`0.038002` vs
    `0.114459` MAE), showing the raw-latent Transformer can learn strong
    within-video structure.
  - 5-feature gated MLP also improves under row split, but less strongly
    (`0.061031` row-100 vs `0.114357` sample MAE), suggesting pooled features
    lose some information needed for fine same-video interpolation.
  - Online, row split is not consistently better than sample split; this
    confirms that row-split offline MAE should not be used as the primary model
    selection metric for deployed adaptive inference.
  - Sample split remains the more relevant offline proxy for generalization,
    but even sample split does not fully predict online target-PSNR control.
- Session log:
  `logs/session_20260630_row_vs_sample_split_readout.md`.
- No commit was made.

## 2026-07-01 Commit Checkpoint

- User requested committing the current workspace state.
- Reviewed `git status`, `git diff --stat`, untracked experiment script sizes,
  and `experiment_results/` symlink targets before committing.
- Confirmed new `experiment_results/` entries are symlinks to `/hy-tmp/...`
  archive roots rather than large video/model files.
- Added this progress note and session log:
  `logs/session_20260701_commit_checkpoint.md`.
- Planned verification before commit:
  - `git diff --check`
  - `python -m py_compile` on modified/new Python modules and experiment
    runners.
