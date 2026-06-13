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
