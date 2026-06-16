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

- Created `experiments/adaptive_threshold_predictor/` as the isolated workspace for prediction-network code.
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
  - `python -m experiments.adaptive_threshold_predictor.inspect_trace_data`
  - `python -m experiments.adaptive_threshold_predictor.train_gate --epochs 1 --batch_size 2 --max_examples 8 --out_dir /hy-tmp/wan22_adaptive_threshold_predictor_smoke`
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
- Added `--feature_set` to `experiments.adaptive_threshold_predictor.train_gate`.
- Added `experiments.adaptive_threshold_predictor.run_feature_ablation` to run all feature sets and write `feature_ablation_summary.json`.
- Smoke validation:
  - `/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m experiments.adaptive_threshold_predictor.run_feature_ablation --epochs 1 --batch_size 2 --max_examples 10 --device cpu --out_root /hy-tmp/wan22_adaptive_threshold_feature_ablation_smoke`
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
  - `python -m py_compile experiments/adaptive_threshold_predictor/models.py experiments/adaptive_threshold_predictor/data.py experiments/adaptive_threshold_predictor/train_gate.py experiments/adaptive_threshold_predictor/inspect_trace_data.py experiments/adaptive_threshold_predictor/run_feature_ablation.py`
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
  - `/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m experiments.adaptive_threshold_predictor.train_gate --feature_set latent_pool --epochs 1 --batch_size 2 --max_examples 20 --device cpu --out_dir /hy-tmp/wan22_adaptive_threshold_train_smoke_v3`

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
  - `/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m experiments.adaptive_threshold_predictor.train_gate --epochs 1 --batch_size 2 --max_examples 30 --device cpu --out_dir /hy-tmp/wan22_adaptive_threshold_candidate_inverse_smoke`

## 2026-06-16 Adaptive Feature Cache And First Ablation

- Raw latent training was too slow because it repeatedly opened 50,000 step `.pt` files.
- Added cached feature support:
  - `experiments/adaptive_threshold_predictor/build_feature_cache.py`
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

- Continued timestep-threshold predictor work under `experiments/adaptive_threshold_predictor/`.
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

- Added control modes to `experiments/adaptive_threshold_predictor/train_gate.py`:
  - `--control_mode condition_only`: use only normalized timestep and normalized PSNR, no latent-derived feature branch.
  - `--control_mode noise_feature`: keep the cached-feature architecture but replace the feature tensor with random noise.
- Added `ConditionOnlyAdaCacheGate` in `experiments/adaptive_threshold_predictor/models.py`.
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
