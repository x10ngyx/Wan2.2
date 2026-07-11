# Progress

This file is intentionally kept concise. It records only:

1. Experiment descriptions, key results, and conclusions.
2. Common errors and their solutions.

Detailed implementation notes, launch commands, intermediate checks, and full tables should live in `reports/`, `logs/`, and archived experiment roots.

## Session Notes

### 2026-07-11 Predictor-RL Speedup-Priority Defaults

- Changed the offline-IQL default `lambda_speedup` from `10.0` to `30.0` in both `predicotr-rl/data.py` and `train_iql.py`, so direct data-builder callers and CLI runs use the same target-speed reward scale. This makes a `0.3x` target-speed miss cost `9` reward units.
- Changed default IQL `beta` from `3.0` to `1.5`. Because advantages are batch-standardized before exponentiation, this reduces the prior extremely concentrated advantage-weighted imitation update while preserving reward-based trajectory preference.
- Retained `lambda_latent=5.0`, `lambda_recompute=0.04`, `lambda_psnr=1.0`, `reuse_cost_ratio=0.081`, `tau=0.7`, `gamma=1.0`, and `rho=0.995`. The preceding scale analysis found these values reasonable or provisionally balanced.
- Updated `predicotr-rl/README.md` to state the speedup-priority objective and numerical rationale. Full GPU training and closed-loop validation remain pending visible GPU access.

### 2026-07-11 Predictor-RL Hyperparameter Scale Analysis

- Quantified the current defaults against all `1000` OpenVid-100 SeaCache candidates. The `reuse_cost_ratio=0.081` final-speedup proxy is well calibrated: absolute error mean `0.0064x`, median `0.0059x`, and max `0.0411x`.
- With default local offsets and `lambda_speedup=10`, terminal speed penalties have median `1.538` and mean `1.768`, versus raw PSNR mean `26.47` and standard deviation `7.78`. This supplies local preference signal but does not strongly enforce low/mid target speeds where the quality cliff is steep.
- Adjacent trajectory break-even analysis: the median speedup weight needed to prefer the faster trace at its faster target is about `30.1` for `0.10 -> 0.15`, `17.2` for `0.15 -> 0.20`, `9.4` for `0.20 -> 0.25`, then below `9` after `0.25 -> 0.30`. Treat `lambda_speedup=10` as a moderate/high-speed default, not a uniformly calibrated target-following weight; run `10/20/30` ablations before changing it.
- The available 2,000-step latent-MSE smoke cache indicates `lambda_latent=5` yields trajectory penalty median `1.02` (mean `1.44`), comparable to the full-trajectory recompute penalty median `0.82` (mean `0.90`) at `lambda_recompute=0.04`; retain these pending full-cache confirmation.
- `tau=0.7`, `gamma=1`, `rho=0.995`, `3e-4` learning rate, and the 30-epoch/256-batch training budget are numerically conventional. The main optimizer concern is batch-standardized advantage with `beta=3`: under a unit-normal approximation it gives `exp(3A)` weights, caps about `6.2%` of samples at `100`, and has mean clipped weight about `12.7`. Ablate `beta=1/1.5/3` (or remove advantage standardization) rather than treating `3` as scale-invariant.
- No predictor code or defaults were changed. GPU remains unavailable, so no full training/closed-loop validation was run.

### 2026-07-11 Predictor-RL Readthrough

- Read the standalone `predicotr-rl/` implementation and its prior session records; no RL, Wan2.2 runtime, checkpoint, dataset, or experiment artifact was changed.
- Confirmed the current scope is an offline-IQL SeaCache timestep policy: historical fixed-threshold traces supply synchronized per-step skip/recompute labels, and deployment is limited to the `SeaCacheRLPolicy` checkpoint loader.
- The train/runtime state is five cached latent feature tensors plus normalized timestep, target speedup, projected full-task speedup, and consecutive-skip count. `action=1` means reuse/skip; `action=0` means recompute.
- No production checkpoint or Wan2.2 cache integration exists. Existing `/hy-tmp/wan22_iql_*` checkpoints are CPU smoke artifacts only. The host currently reports no visible NVIDIA GPU, so no full latent-MSE preparation, GPU training, or closed-loop quality/speed evaluation was run.

### 2026-07-10 Predictor-RL Framework Review

- Reviewed `predicotr-rl/` from method, implementation, and default-parameter perspectives; no training, inference, or Wan2.2 runtime code was changed.
- Main assessment: the current directory is a useful offline-IQL prototype, but it should not yet be treated as a validated adaptive predictor because it learns from fixed SeaCache threshold trajectories and has no closed-loop Wan2.2 evaluation.
- Key concerns: reward scale is dominated by raw PSNR versus small target-speedup penalties, checkpoint selection uses behavior-action accuracy instead of speed/quality objective performance, advantage normalization changes standard IQL/AWAC weighting semantics, target-speedup augmentation is local and weak, and full bundle construction can be slow without cached latent-MSE/state preprocessing.

### 2026-07-10 RL4Acc Terminal Speedup Proxy Alignment

- Updated `predicotr-rl/data.py` so the terminal speedup penalty uses the final action-cost proxy computed from the full skip/recompute sequence, instead of using measured `summary.csv` speedup.
- Rationale: `Speedup_current` in the state and `R_terminal` speedup penalty now share the same proxy dynamics; measured speedup remains stored as `achieved_speedups` for analysis.
- Updated `predicotr-rl/README.md` to document `final_proxy_speedup` in the terminal reward.
- Validation: `python -m py_compile predicotr-rl/data.py predicotr-rl/train_iql.py predicotr-rl/policy.py predicotr-rl/models.py` passed.

### 2026-07-10 RL4Acc Method Sanity Review

- Reviewed `doc/RL4Acc.pdf` with `pypdf` text extraction and compared it against the current `predicotr-rl/` implementation.
- Assessment: the IQL update formulas, binary skip action, state scalars, latent-MSE immediate reward, recompute penalty, and terminal PSNR/speedup reward are broadly aligned with the PDF.
- Main caveat: this should be described as a faithful offline-IQL prototype based on existing SeaCache traces, not a strict executable reproduction of an online denoising environment. The implementation reconstructs transitions from fixed threshold trajectories, duplicates each trajectory over local target-speedup offsets around its measured speedup, uses a calibrated speedup proxy, and has no closed-loop Wan2.2 runtime evaluation yet.
- Method risks to track: limited action coverage from fixed threshold traces, possible distribution shift when the learned policy composes skip/recompute paths not present in data, reward-scale balance between PSNR and latent/recompute/speedup penalties, proxy-vs-real speedup mismatch, and checkpoint selection by validation policy accuracy rather than target PSNR/speedup performance.

### 2026-07-10 RL4Acc Predictor-RL Formula Audit

- Audited `doc/RL4Acc.pdf` against the actual implementation directory `predicotr-rl/` (the workspace does not currently contain `predictor-rl/`).
- Conclusion: the current implementation follows the PDF's high-level offline-IQL method and the main math formulas for state layout, binary skip action, latent-MSE/recompute immediate reward, terminal PSNR/speedup reward, V expectile loss, double-Q Bellman update, and advantage-weighted policy update.
- Not identical in execution semantics: transitions are reconstructed from fixed precomputed SeaCache traces rather than produced by an online denoising environment; each trace is duplicated over local target-speedup offsets around its measured speedup; the PDF's `z_{t-1}` reward term is concretely implemented by comparing stored `step_{t+1}` candidate/baseline latents and final `trace_done.pt` latents; and the implementation uses sampled traces rather than one threshold group per prompt.
- PDF-unspecified implementation choices include the five concrete latent feature sets, cond/uncond action unioning, full-task speedup proxy with calibrated reuse cost, concrete MLP architecture/optimizer/hyperparameters, state normalization, sample-level train/val split, checkpoint/export format, policy probability thresholding, cached latent-MSE files, and logging/evaluation metrics.

### 2026-07-10 RL4Acc Current-Code Recheck

- Rechecked `doc/RL4Acc.pdf` against the current `predicotr-rl/` source. This supersedes the earlier conformance note for the older intermediate version because the current code now includes the PDF-style latent-MSE immediate reward, terminal speedup-target penalty, and 4-scalar state layout.
- Current explicit mismatches are mostly about execution/data semantics rather than IQL formulas: training reconstructs transitions from precomputed SeaCache trace rows instead of running the PDF's denoising transition online; each trace is duplicated over local target-speedup offsets around its measured speedup; the implementation consumes fixed SeaCache trace candidates rather than the PDF's simpler "one threshold group per prompt" data statement.
- Current PDF-unspecified additions include five concrete latent feature sets, cond/uncond action unioning, current-speedup proxy cost model, target-Q soft updates, AdamW/MLP architecture details, state normalization, sample-level splits, checkpoint/export format, policy probability thresholding, latent-MSE cache, and training/evaluation logging.
- No Wan2.2 runtime code, RL implementation code, checkpoints, datasets, or experiments were changed during this recheck.

### 2026-07-10 RL4Acc Implementation Conformance Check

- Compared `doc/RL4Acc.pdf` against the actual implementation directory `predicotr-rl/` (note spelling; `predictor-rl/` does not exist in the current workspace).
- Assessment: the implementation follows the PDF's high-level offline IQL framework, including state-conditioned binary skip policy, V/Q/policy MLPs, expectile V update, double-Q Bellman update, and advantage-weighted policy update with batch-normalized advantage.
- Not fully identical to the PDF formulas: reward was simplified to recompute penalty plus terminal normalized PSNR; the PDF's latent MSE immediate term and terminal achieved-vs-target speedup penalty are not implemented. The PDF's state vector was extended with reuse ratio and remaining-step fraction, and state transitions are reconstructed from fixed trace rows rather than simulated by the denoising dynamics described in the proposal.
- No Wan2.2 runtime code, RL implementation code, checkpoints, datasets, or experiments were changed during this conformance check.

### 2026-07-10 Predictor RL IQL Framework

- Added standalone `predicotr-rl/` prototype for offline IQL training of adaptive SeaCache skip/recompute decisions. The implementation keeps cond/uncond decisions synchronized at one action per denoising step and uses the existing five latent feature sets (`latent_pool`, `temporal_mean`, `temporal_var`, `frame_diff_mean`, `frame_diff_var`) as `Z_t`.
- Implemented feature-cache dataset construction from OpenVid-100 SeaCache traces, synchronized action parsing from `seacache_skipping_path`, scalar progress features, V/Q/policy MLPs, expectile V loss, double-Q Bellman updates, advantage-weighted policy loss, checkpoint saving, and a lightweight `SeaCacheRLPolicy` loader.
- State construction was tightened to match `doc/RL4Acc.pdf` exactly: `[timestep, Z_t, Speedup_target, Speedup_current, C_t]`. The earlier extra `reuse_ratio_so_far` and `remaining_steps_norm` helper scalars were removed. `Speedup_current` now uses full-task accounting: completed steps use observed reuse/recompute proxy cost, while unfinished steps are counted as full recompute cost.
- Calibrated the `Speedup_current` proxy against OpenVid-100 measured speedup. The previous `reuse_cost_ratio=0.05` overestimated final speedup with about `5.01%` MAPE; fitted `reuse_cost_ratio=0.081` reduces proxy MAPE to about `0.32%`, so the default was updated to `0.081`.
- IQL `best_model.pt` selection now uses maximum validation `policy_accuracy` instead of minimum validation `pi_loss`; `pi_loss` remains logged as a training diagnostic.
- Updated reward construction to match `doc/RL4Acc.pdf`: immediate reward now includes `-lambda_latent * MSE(z_{t-1}, z_gt_{t-1})` from paired SeaCache/baseline step latents plus recompute penalty, and terminal reward now uses raw PSNR minus `lambda_speedup * abs(final_proxy_speedup - target_speedup)`. Each offline trajectory is duplicated over local target-speedup offsets so the speedup penalty has nonzero but behavior-near training signal.
- Target-speedup augmentation was changed from a global fixed grid for every trajectory to local offsets around each trajectory's measured speedup. Default offsets are `[-0.3, -0.15, 0, 0.15, 0.3]` with targets clipped to `[1.0, 4.0]`, so a `2.4x` trajectory uses targets about `2.1/2.25/2.4/2.55/2.7`.
- The latent MSE target for step `0..48` uses `step_{i+1}.pt['latent']`; for final step `49`, it now uses `trace_done.pt['final_latent']` instead of falling back to `step_049.pt`.
- Added `predicotr-rl/prepare_data.py` for precomputing `latent_mse_to_baseline.pt`, but full latent-MSE preprocessing was not run to completion because the current session has no visible GPU and CPU-only `.pt` latent loading/MSE computation is too slow for the full 50,000-row cache. Training code is ready; full strict-reward training should wait for GPU or a faster preprocessing path.
- Reward defaults were rescaled after checking term magnitudes: `lambda_latent=5.0`, `lambda_recompute=0.04`, `lambda_psnr=1.0`, and `lambda_speedup=10.0`.
- Validation: Python compile passed for all new files; CPU smoke training with `max_examples=2000`, one epoch, and `--torch_threads 1` produced `best_model.pt`/`final_model.pt`; policy-loader smoke returned a skip probability/action from the smoke checkpoint.
- Strict-reward validation: CPU smoke training with `max_examples=80`, target speedups `1.5 2.5`, and a temporary latent-MSE cache completed one epoch and produced checkpoints.
- Runtime integration into Wan2.2 inference was intentionally not changed in this step; the new directory provides the RL-trained network and loader for later replacement of the previous predictor network.

### 2026-07-10 RL4Acc Predictor RL Proposal Review

- Read `doc/RL4Acc.pdf`, a 3-page CodiMD proposal for training an offline IQL policy that chooses per-denoising-step skip/recompute actions conditioned on timestep, latent features, target speedup, current achieved speedup, and consecutive-skip count.
- Assessment: the RL framing matches the sequential nature of cache decisions better than fixed-threshold inverse prediction, but the current OpenVid-100 fixed-threshold trajectories are likely too narrow for stable offline RL unless the action coverage is enriched with multiple diverse trajectories per prompt or a simulator/replay environment.
- No Wan2.2 code, experiments, datasets, predictor checkpoints, or cache logic were changed.

### 2026-07-08 OpenVid-100 HF Dataset OSS Bundle

- Prepared a Hugging Face-ready dataset layout from `/hy-tmp/openvid_100_seacache_trace_data` without exposing the original multi-machine shard layout. Public paths use `prompt_001` through `prompt_100` and `threshold_0p10` through `threshold_0p80`.
- Staging root: `/hy-tmp/hf_staging/wan22_openvid100_seacache_trace`.
- OSS transfer bundle root: `/hy-tmp/oss_upload/wan22_openvid100_seacache_trace_20260708`, size about `134G`, containing `27` tar archives plus `HOW_TO_UPLOAD_TO_HUGGINGFACE.md`, `archive_manifest.json`, and `checksums/archive_sha256s.txt`.
- Dataset indexes include `100` prompts, `1000` candidate rows, and `50000` step-level training rows split into `10` Parquet shards plus JSONL mirrors. Each training row includes prompt/threshold/step identifiers, reuse/recompute decision, branch keys, cache metrics, raw latent `.pt` path, baseline/candidate video paths, logs, PSNR, elapsed time, and speedup.
- Validation completed: staging `scripts/verify_dataset.py` passed, and `sha256sum -c checksums/archive_sha256s.txt` passed for all archives. One source PSNR text log was missing/NaN for `prompt_034`, `threshold_0p10`; PSNR JSON and summary metrics are present.
- OSS upload itself was not run because this machine currently has no `ossutil/ossutil64` command configured and no bucket/endpoint/prefix was provided.

### 2026-07-08 Q-learning Concept Explanation

- User asked for an example-based explanation of Q-learning.
- No Wan2.2 code, experiments, datasets, or cached artifacts were changed.

## Experiment Results

### Fixed ZEUS Timestep Cache, Ali-10, DPM++

- Root: `/hy-tmp/wan22_zeus_timestep_cache_50step_45f_480p_full_20260608_114307`
- Description: fixed ZEUS-style timestep-output reuse on 10 Ali prompts.
- Results: overall speedup `1.986x`, mean FFmpeg PSNR `23.705 dB`, no failures.
- Conclusion: fixed ZEUS gives a strong speedup but quality is only moderate; useful as an early timestep-cache baseline.

### ZEUS-Threshold Timestep Cache, Ali Prompt 1/2, DPM++

- Main root: `/hy-tmp/wan22_zeus_threshold_reuse_interp_10prompt_5th_20260608_195427`
- Description: latent relative-L1 threshold controls whole denoiser-output reuse.
- Results: prompt-01 threshold `0.005` had higher PSNR but modest speedup; thresholds `0.02+` improved speed but fell to about `18.6-18.9 dB` PSNR.
- Conclusion: ZEUS-threshold is sensitive to threshold choice and is dominated by SeaCache on later prompt-01/02 comparisons.

### Original Three-Cache Threshold Grid, Ali Prompt 1, DPM++

- Root: `/hy-tmp/wan22_three_cache_threshold_grid_prompt01_50step_45f_480p_20260610_012518`
- Description: 64 combinations of timestep ZEUS-threshold, original block-group cache, and original CFG threshold cache.
- Results: completed `64/64`, no failures. Fastest candidate `ts_0p6__bg_1__cfg_1`: `4.080x`, PSNR `15.225 dB`. Best finite high-PSNR candidate above `25 dB`: `ts_0p005__bg_0p001__cfg_0p001`: `1.039x`, PSNR `26.954 dB`. Best speed at PSNR `>=22 dB`: `1.204x`, PSNR `23.448 dB`. Best speed at PSNR `>=20 dB`: `1.369x`, PSNR `20.042 dB`.
- Conclusion: combining the original caches produced high-speed low-quality points, but useful high-quality acceleration was limited.

### Cache Ablation, Ali Prompt 1, DPM++

- Root: `/hy-tmp/wan22_cache_ablation_prompt01_50step_45f_480p_20260609_184625`
- Description: isolated and paired cache ablations using the original cache variants.
- Results: baseline compute `522.603s`. `timestep_only`: `1.600x`, `18.606 dB`; `block_only`: `1.362x`, `19.396 dB`; `cfg_only`: `1.148x`, `21.571 dB`; `timestep_block`: `1.748x`, `18.159 dB`; `timestep_cfg`: `1.332x`, `20.910 dB`; `block_cfg`: `1.352x`, `19.446 dB`; `all_three`: `1.370x`, `19.603 dB`.
- Conclusion: timestep cache contributed the largest speedup, CFG cache was more quality-preserving, and naive three-cache combination did not guarantee the best frontier.

### SeaCache Timestep-Only, Ali Prompt 1/2, DPM++

- Prompt-01 root: `/hy-tmp/wan22_seacache_50step_45f_480p_20260611_191733`
- Prompt-02 roots: `/hy-tmp/wan22_seacache_prompt02_dense_20260611_204826`, `/hy-tmp/wan22_seacache_prompt02_highthr_20260612_000218`
- Description: timestep-only SeaCache threshold sweep.
- Prompt-01 results: threshold `0.10`: `1.112x`, `36.303 dB`; `0.20`: `1.569x`, `24.558 dB`; `0.30`: `1.966x`, `20.562 dB`; `0.50`: `2.779x`, `19.460 dB`.
- Prompt-02 results: threshold `0.10`: `1.090x`, `45.532 dB`; `0.20`: `1.562x`, `30.097 dB`; `0.30`: `1.965x`, `29.582 dB`; `0.50`: `2.641x`, `23.725 dB`; `0.60`: `3.098x`, `20.262 dB`; `0.80`: `3.499x`, `18.631 dB`.
- Conclusion: SeaCache dominated ZEUS-threshold on the observed prompt-01/02 speed-quality frontier.

### OpenVid SeaCache Local Shard, Prompts 76-100, DPM++

- Root: `/hy-tmp/wan22_seacache_openvid100_50step_45f_480p_20260612_002814`
- Description: timestep-only SeaCache sweep on 25 OpenVid prompts, thresholds `0.10` through `0.80`.
- Results: completed `25/25` baselines and `250/250` candidates, no failures. Aggregate mean PSNR/speedup by threshold: `0.10`: `42.333 dB`, `1.113x`; `0.15`: `34.222 dB`, `1.412x`; `0.20`: `30.188 dB`, `1.575x`; `0.25`: `26.787 dB`, `1.844x`; `0.30`: `25.170 dB`, `1.976x`; `0.40`: `22.836 dB`, `2.418x`; `0.50`: `21.429 dB`, `2.746x`; `0.60`: `19.567 dB`, `3.112x`; `0.70`: `19.282 dB`, `3.337x`; `0.80`: `19.004 dB`, `3.517x`.
- Conclusion: the OpenVid shard establishes a smooth SeaCache threshold-to-quality/speed curve and supplies the main data source for early adaptive-threshold work.

### Sea CFG Cache vs Original CFG Cache, Ali Prompt 1, DPM++

- Root: `/hy-tmp/wan22_cfg_cache_prompt01_50step_45f_480p_20260613_163243`
- Description: CFG-only comparison between original `--cfg_cache threshold` and SeaCache-style `--cfg_cache sea-threshold`.
- Results: old CFG `0.02`: `1.041x`, `26.732 dB`; old CFG `0.03`: `1.137x`, `21.571 dB`; Sea CFG `0.10`: `1.007x`, `37.457 dB`; Sea CFG `0.20`: `1.175x`, `26.226 dB`; Sea CFG `0.30`: `1.297x`, `21.359 dB`.
- Conclusion: Sea CFG gives better quality at comparable reuse. `0.20` is the useful mid-point; `0.30` is aggressive.

### Sea Timestep + Sea CFG, Ali Prompt 1, DPM++

- Root: `/hy-tmp/wan22_timestep_cfg_prompt01_no_uncond_skip_accum_50step_45f_480p_20260613_213000`
- Description: sea timestep cache plus Sea CFG cache with No-Skip-Accum behavior for skipped uncond branches.
- Results: `ts=0.10,cfg=0.10`: `1.067x`, `36.747 dB`; `0.10,0.20`: `1.256x`, `26.430 dB`; `0.20,0.10`: `1.498x`, `24.433 dB`; `0.20,0.20`: `1.550x`, `24.848 dB`.
- Conclusion: chosen No-Skip-Accum behavior is maintained. The combination is close to SeaCache-only `0.20` but does not clearly beat it on speed.

### Sea Block Cache vs Original Block-Group Cache, Ali Prompt 1, DPM++

- Report: `reports/report_block_cache_sea_vs_old_prompt01.md`
- Description: original block-group metric versus Sea-style block-group metric `sea_full_rel_l1` with accumulated decision mode.
- Results: full tables are in the report; Sea-style block cache was evaluated against `/hy-tmp/wan22_block_group_sea_full_prompt01_50step_45f_480p_20260614_235605`.
- Conclusion: Sea-style block metric is the preferred block-cache experiment path, but it has higher GPU memory pressure because it stores full filtered features.

### Sea-Style Three-Cache Grid, Ali Prompt 1, DPM++

- Root: `/hy-tmp/wan22_three_cache_sea_prompt01_50step_45f_480p_20260614_005404`
- Description: 125 combinations of SeaCache timestep, Sea-style block-group, and Sea CFG cache.
- Results: completed `125/125`, no failures. Fastest finite candidate: `sea_ts_1p00__sea_bg_1p00__sea_cfg_1p00`, `5.644x`, `11.914 dB`. Best finite PSNR: `sea_ts_0p05__sea_bg_0p10__sea_cfg_0p05`, `0.987x`, `37.465 dB`. Best speed by PSNR target: `>=35 dB`: `1.025x`; `>=26 dB`: `1.208x`; `>=24 dB`: `1.496x`; `>=19 dB`: `2.845x`; `>=18 dB`: `3.575x`; `>=15 dB`: `4.873x`.
- Conclusion: Sea-style three-cache completed without OOM and improved the quality ceiling versus the old three-cache grid, but moderate thresholds around `0.10-0.20` remain the useful range; aggressive thresholds quickly degrade quality.

### VBench10 Sea-Style Three-Cache and Timestep-Only SeaCache, DPM++

- Root: `/hy-tmp/wan22_vbench10_three_cache_full_merge_and_timestep_only_full_20260623`
- Description: imported full VBench10 SeaCache/timestep-only and three-cache Sea-style result package.
- Results: package integrity check found `760` MP4 files, `26` CSV files, `1532` JSON files, and `0` failed files.
- Conclusion: this is the canonical VBench10 SeaCache result source for later ZEUS/SeaCache, sampler, and adaptive comparisons.

### ZEUS VBench10, DPM++

- Root: `/hy-tmp/wan22_zeus_vbench10_50step_45f_480p_20260624_003030`
- Description: fixed ZEUS and ZEUS-threshold on VBench10, reusing existing baselines.
- Results: completed `10/10` fixed-ZEUS and `50/50` ZEUS-threshold candidates, no failures. Fixed ZEUS: `2.021x`, mean PSNR `23.996 dB`. ZEUS-threshold: `0.005`: `1.129x`, `23.020 dB`; `0.02`: `1.604x`, `20.868 dB`; `0.08`: `2.282x`, `20.690 dB`; `0.20`: `2.648x`, `20.707 dB`; `0.60`: `2.793x`, `20.734 dB`.
- Conclusion: on VBench10, ZEUS-threshold speed rises with threshold but PSNR saturates around `20.7 dB`; fixed ZEUS is the more balanced ZEUS point.

### SeaCache UniPC, Ali-10 and VBench10

- Ali-10 root: `/hy-tmp/wan22_seacache_unipc_ali10_50step_45f_480p_20260627_023222`
- VBench10 root: `/hy-tmp/wan22_seacache_unipc_vbench10_50step_45f_480p_20260627_023222`
- Description: SeaCache timestep-only runs using UniPC instead of DPM++, thresholds `0.10`, `0.20`, `0.30`, `0.50`.
- Results: full VBench10 comparison against DPM++ exists. Full Ali-10 UniPC exists, but full Ali-10 DPM++ SeaCache does not; only prompt-01/02 DPM++ pilots exist.
- Conclusion: sampler comparison is complete for VBench10 only. Do not claim full Ali-10 DPM++ versus UniPC SeaCache comparison until the missing Ali-10 DPM++ run is done.

### AdaCache Wan2.2 VBench Smoke and Reproduction Import

- Smoke root: `/hy-tmp/wan22_adacache_vbench_smoke_20260616_1908`
- Reproduction root: `/hy-tmp/adacache_wan22_vbench10_reproduction_20260623`
- Description: official-style AdaCache adapter and VBench10 reproduction package.
- Smoke result: baseline completed (`533.455s` compute), but the full official-style AdaCache candidate OOMed on A100 80GB at step `0` because it cached residuals for all blocks and both CFG branches.
- Imported reproduction result: AdaCache slow mean speedup `1.545x`, mean PSNR `23.561 dB`; AdaCache fast mean speedup `2.702x`, mean PSNR `18.635 dB`; no failures in imported package.
- Conclusion: all-block AdaCache does not fit the local single A100 80GB default Wan2.2 setting without memory reduction. Use imported reproduction results for comparison, or implement selected-block/offloaded cache variants before local full-size reruns.

### TaylorSeer Wan2.2 Standalone Integration

- Location: `third_party/taylorseer_wan22/`
- Description: standalone TaylorSeer-style Wan2.2 adapter, separated from main Wan cache code.
- Results: only static/CPU validation was completed. No full GPU quality/speed result is recorded in this file.
- Conclusion: TaylorSeer code path is available for future experiments but should not be treated as measured on Wan2.2 until a full archived run is completed.

### OpenVid-100 Trace Data Import and Layout

- Root: `/hy-tmp/openvid_100_seacache_trace_data`
- Description: extracted OpenVid prompt archives and organized a flat symlink-based training layout under `data/`.
- Results: `100` baselines, `1000` SeaCache videos, per-threshold summaries, `1000` candidate rows, no broken links in the flat layout. Extracted size about `135G`.
- Conclusion: this is the main training data root for adaptive threshold predictor work.

### Adaptive Threshold Predictor, Early MLP Feature Ablations

- Feature cache: `/hy-tmp/wan22_adaptive_threshold_feature_cache_candidate_inverse_20260616_012409`
- Description: cached 2x2x2 pooled features for `candidate_inverse` threshold prediction.
- Results: 50,000 examples, five feature sets, feature dim `128`, cache size about `124M`. Best short 3-epoch validation loss was `2x2x2 temporal_mean`: val loss `0.012259`, MAE `0.120107`; `latent_pool` had MAE `0.116558`.
- Conclusion: timestep+PSNR explains much of the label structure, but latent-derived features improve validation loss over condition-only/noise controls by about `13-16%`.

### Adaptive Predictor Pooling Grid and Capacity Ablations

- Grid root: `/hy-tmp/wan22_adaptive_threshold_grid_ablation_20260616_020314`
- Capacity roots: `/hy-tmp/wan22_adaptive_threshold_feature_ablation_long_20260616`, `/hy-tmp/wan22_adaptive_threshold_feature_ablation_hdim16_20260616`, `/hy-tmp/wan22_adaptive_threshold_feature_ablation_hdim8_20260616`
- Description: compared pooling grids and hidden dimensions for the early MLP predictor.
- Results: larger grids did not improve best validation loss. Hidden dim `16` gave the best tested single-split loss for temporal_mean (`0.012254`) with much fewer parameters than hidden dim `64`; hidden dim `8` underfit.
- Conclusion: early MLP predictor overfits quickly; use early stopping. Hidden dim `16` + `temporal_mean` was the best early lightweight baseline.

### Adaptive SeaCache, Ali Prompt 1/2, Early MLP Predictor

- Root: `/hy-tmp/wan22_adaptive_seacache_ali_prompt12_50step_45f_480p_20260616_165412`
- Description: timestep-only adaptive SeaCache using the hidden-dim-16 temporal_mean MLP predictor.
- Results: six candidates completed, no failures. Ali-001 target `20/25/30`: `2.870x/1.869x/1.543x`, PSNR `19.325/19.450/24.462 dB`. Ali-002 target `20/25/30`: `3.051x/2.270x/1.641x`, PSNR `20.288/26.998/29.354 dB`.
- Conclusion: predicted thresholds drive SeaCache in the expected direction but do not dominate fixed-threshold SeaCache. Ali-001 target 25 is clearly dominated by fixed threshold points.

### Adaptive SeaCache Train15/Test5 and Overhead, Early MLP Predictor

- Train/test root: `/hy-tmp/wan22_adaptive_seacache_train15_test5_50step_45f_480p_20260619_135521`
- Overhead root: `/hy-tmp/wan22_adaptive_seacache_overhead_train5_50step_45f_480p_20260619_143632`
- Description: larger OpenVid adaptive SeaCache run plus online/replay overhead measurement.
- Results: original runs were interrupted by OOM before full completion. Partial train15/test5 completed 12/60 candidates; overhead completed 11/15 online/replay pairs. Partial train15/test5 means over completed train prompts: target 20 `3.226x`, `23.544 dB`; target 25 `2.575x`, `24.928 dB`; target 30 `2.066x`, `29.740 dB`. Predictor timing overhead was about `0.1%` of online compute.
- Conclusion: predictor call overhead is negligible, but the early runner had cache-lifecycle OOM issues and partial results should not be treated as final full-run metrics.

### Fixed SeaCache Control, OpenVid Train15/Test5, DPM++

- Root: `/hy-tmp/wan22_seacache_train15_test5_50step_45f_480p_20260623_160513`
- Description: fixed-threshold SeaCache control on the same 20 OpenVid prompts as adaptive train15/test5.
- Results: completed `80/80`, no failures. Threshold `0.1`: `1.138x`, `42.861 dB`; `0.2`: `1.607x`, `30.548 dB`; `0.4`: `2.467x`, `23.936 dB`; `0.6`: `3.176x`, `21.229 dB`.
- Conclusion: fixed SeaCache matches or beats the early adaptive predictor at comparable operating points; adaptive target control was not yet superior.

### MiniDiT-CLS Adaptive Threshold Predictor, Offline

- Sample split root: `/hy-tmp/wan22_adaptive_threshold_mini_dit_cls_convpatch_3x12x8_d96_l2_bs128_20260629_214906`
- Row split root: `/hy-tmp/wan22_adaptive_threshold_mini_dit_cls_convpatch_rowsplit_packed_d96_l2_bs128_20260629_232659`
- Description: raw-latent Conv3d patch MiniDiT predictor with CLS readout, about `724,513` parameters.
- Results: sample split best validation MAE about `0.114459`; row split best validation MAE `0.0380019`.
- Conclusion: MiniDiT can fit same-video/row-level structure very well, but sample-split generalization remains weak. Row-split metrics should not be used alone for deployment decisions.

### 5-Feature Gated MLP Adaptive Threshold Predictor, Offline

- Range-constrained roots:
  - sample split: `/hy-tmp/wan22_adaptive_threshold_mlp_gated_5feature_range_samplesplit_20260630_035000`
  - row split 30: `/hy-tmp/wan22_adaptive_threshold_mlp_gated_5feature_range_rowsplit_gpu_20260630_035000`
  - row split 100: `/hy-tmp/wan22_adaptive_threshold_mlp_gated_5feature_range_rowsplit_gpu_long100_20260630_035000`
- Description: gated fusion over `latent_pool`, `temporal_mean`, `temporal_var`, `frame_diff_mean`, and `frame_diff_var`, with threshold output constrained to `[0.10,0.80]`; `83,526` parameters.
- Results: sample split best MAE `0.114357`; row split 30 best MAE `0.077050`; row split 100 best MAE `0.061031`.
- Conclusion: much smaller than MiniDiT and sample-split MAE is essentially tied, but row-split performance is clearly worse than MiniDiT. Longer row-split training matters; sample split early-stops.

### Adaptive Predictor Speedup-Condition Input

- Description: added `target_speedup` as a third condition input alongside timestep and target PSNR for the 5-feature gated MLP and MiniDiT/Transformer predictor paths; removed `target_oracle` from the training/cache-building pipeline.
- Results: OpenVid-100 trace summary has finite speedup for all `1000/1000` candidate rows, range `1.092x-3.568x`. Static compile passed; raw and cached datasets expose `target_speedup`; CPU smoke training passed for both 5-feature gated MLP and MiniDiT, with metrics grouped by `train_by_target_speedup`.
- Conclusion: the predictor pipeline is ready for speedup-conditioned retraining. Existing two-condition checkpoints are incompatible with the new three-condition input and should be retrained.

### Speedup-Conditioned Adaptive Predictor Retraining

- Aggregate root: `/hy-tmp/wan22_speedup_condition_retrain_20260706_171523`
- Description: retrained MiniDiT/Transformer and 5-feature gated MLP predictors with `target_speedup` as a third condition input, preserving the previous transformer training settings and result-recording convention except for the added speedup condition. Both sample-split and row-split were trained; 5-feature runs used a 100-epoch budget.
- Results: MiniDiT sample split root `/hy-tmp/wan22_adaptive_threshold_mini_dit_cls_convpatch_3x12x8_d96_l2_bs128_speedup_20260706_171523`, best epoch `13`, best val MAE `0.005919`, early stopped at epoch `18`. MiniDiT row split root `/hy-tmp/wan22_adaptive_threshold_mini_dit_cls_convpatch_rowsplit_packed_d96_l2_bs128_speedup_20260706_171523`, best epoch `30`, best val MAE `0.003765`. 5-feature sample split root `/hy-tmp/wan22_adaptive_threshold_mlp_gated_5feature_range_speedup_samplesplit_long100_20260706_171523`, best epoch `52`, best val MAE `0.008678`, early stopped at epoch `72`. 5-feature row split root `/hy-tmp/wan22_adaptive_threshold_mlp_gated_5feature_range_speedup_rowsplit_long100_20260706_171523`, best epoch `93`, best val MAE `0.007119`.
- Artifacts: all four runs produced `best_model.pt`, `best_model_checkpoint.pt`, `final_model.pt`, `final_model_checkpoint.pt`, `val_predictions.csv`, `epoch_metrics.csv`, `epoch_metrics.jsonl`, `config.json`, `split.json`, `model_summary.json`, and `metrics.json`; symlinks were added under `experiment_results/`.
- Conclusion: adding measured speedup makes the offline fixed-candidate inverse task much easier than the prior two-condition setup. These metrics should still be treated as offline inverse-task results; online adaptive SeaCache validation with the new checkpoints is still required before deployment claims.

### Condition-Only Speedup Baseline Check

- Aggregate root: `/hy-tmp/wan22_condition_only_speedup_ablation_rowsplit_20260706_221900`
- Description: checked the two condition-only baseline plans requested after the speedup-conditioned retraining, using row split and the 5-feature long100 training settings but with no latent/features consumed by the model. The full condition-only run used `(timestep, target_psnr, target_speedup) -> threshold`; the speedup-only run used `(timestep, target_speedup) -> threshold`.
- Results: full-condition root `/hy-tmp/wan22_adaptive_threshold_condition_only_fullcond_rowsplit_long100_20260706_221900`, `12,929` parameters, best epoch `60`, best val MAE `0.007347`, early stopped at epoch `80`. Speedup-only root `/hy-tmp/wan22_adaptive_threshold_condition_only_speeduponly_rowsplit_long100_20260706_221900`, `12,865` parameters, best epoch `85`, best val MAE `0.007612`, completed `100` epochs. Both runs produced complete final artifacts including checkpoints, metrics, and `val_predictions.csv`.
- Conclusion: condition-only performance is essentially tied with the 5-feature speedup-conditioned row-split model (`0.007119`), and removing target PSNR only slightly worsens MAE. Most of the offline inverse-task gain therefore comes from measured speedup, which nearly determines the fixed SeaCache threshold in this dataset.

### Speedup-Conditioned MiniDiT Row-Split Online Test

- Root: `/hy-tmp/wan22_adaptive_seacache_mini_dit_rowsplit_speedup_sweep_50step_45f_480p_20260706_194715`
- Description: launched the row-split MiniDiT/Transformer online adaptive SeaCache test using the speedup-conditioned checkpoint, preserving the prior MiniDiT split-compare prompt setting: VBench10 first 3 prompts plus OpenVid train first 3 prompts, target PSNRs `22` and `28`, seed `42`, `832*480`, `45` frames, `50` DPM++ steps. Only `row_split` is tested. Target speedup is swept in three settings per PSNR: PSNR `22` uses `2.2`, `2.5`, `2.8`; PSNR `28` uses `1.4`, `1.7`, `2.0`, where the middle value is the expected speedup estimated from fixed SeaCache/previous online curves and the other two are lower/higher probes.
- Status: completed `36/36`, no failed rows; tmux session is no longer running.
- Artifacts: symlinked under `experiment_results/wan22_adaptive_seacache_mini_dit_rowsplit_speedup_sweep_50step_45f_480p_20260706_194715`; launch log `/hy-tmp/wan22_adaptive_seacache_mini_dit_rowsplit_speedup_sweep_50step_45f_480p_20260706_194715.tmux.log`.
- Conclusion: final aggregate results are recorded below in the Speedup-Conditioned MiniDiT Online Sweep section and in `reports/report_predictor_speedup.md`.

### MiniDiT Online Adaptive SeaCache Split Compare

- Root: `/hy-tmp/wan22_adaptive_seacache_mini_dit_split_compare_50step_45f_480p_20260630_025328`
- Description: 24-candidate online adaptive SeaCache comparison: sample vs row split, VBench10 vs OpenVid train, targets `22` and `28`, three prompts per dataset.
- Results: completed `24/24`, no failures. Aggregates: VBench10 sample target22 `2.113x`, `16.737 dB`; VBench10 sample target28 `1.582x`, `23.796 dB`; VBench10 row target22 `2.068x`, `20.466 dB`; VBench10 row target28 `1.539x`, `25.469 dB`; OpenVid sample target22 `2.598x`, `22.151 dB`; OpenVid sample target28 `1.794x`, `29.019 dB`; OpenVid row target22 `2.447x`, `23.007 dB`; OpenVid row target28 `1.633x`, `27.710 dB`.
- Conclusion: row split helps VBench target calibration versus sample split but still undershoots VBench targets. OpenVid train calibration is better. Fixed-threshold SeaCache remains competitive or better on VBench10 target closeness.

### 5-Feature Gated MLP Online Adaptive SeaCache Split Compare

- Root: `/hy-tmp/wan22_adaptive_seacache_mlp_gated_5feature_range_split_compare_50step_45f_480p_20260630_050727`
- Description: same 24-candidate online protocol as MiniDiT, using 5-feature gated MLP checkpoints.
- Results: completed `24/24`, no failures. Aggregates: VBench10 sample target22 `2.229x`, `17.159 dB`; VBench10 sample target28 `1.539x`, `25.460 dB`; VBench10 row target22 `2.065x`, `20.810 dB`; VBench10 row target28 `1.627x`, `24.484 dB`; OpenVid sample target22 `2.559x`, `22.858 dB`; OpenVid sample target28 `1.718x`, `27.365 dB`; OpenVid row target22 `2.523x`, `21.970 dB`; OpenVid row target28 `1.773x`, `26.095 dB`.
- Conclusion: online results are mixed rather than uniformly worse than MiniDiT. Across 24 candidates, both current predictors hit target PSNR in only `8/24` cases. The `candidate_inverse` offline task remains poorly aligned with online target control.

### MiniDiT vs Fixed SeaCache, VBench10 Three-Prompt Check

- Adaptive source: `/hy-tmp/wan22_adaptive_seacache_mini_dit_split_compare_50step_45f_480p_20260630_025328/results/summary.csv`
- Fixed source: `/hy-tmp/wan22_vbench10_three_cache_full_merge_and_timestep_only_full_20260623/timestep_only_seacache_vbench10_full/wan22_seacache_vbench10_50step_45f_480p_20260618_161845/merged/summary.csv`
- Description: compare MiniDiT row-split adaptive SeaCache against fixed SeaCache on the same VBench10 prompts.
- Results: target 22 adaptive `2.068x`, `20.466 dB`; closest fixed threshold `0.30`: `1.979x`, `22.218 dB`. Target 28 adaptive `1.539x`, `25.469 dB`; closest fixed threshold `0.15`: `1.410x`, `27.728 dB`.
- Conclusion: MiniDiT row-split is slightly faster but undershoots target PSNR more than fixed SeaCache threshold selection; it does not beat fixed threshold on target closeness.

### Speedup-Conditioned MiniDiT Online Sweep

- Root: `/hy-tmp/wan22_adaptive_seacache_mini_dit_rowsplit_speedup_sweep_50step_45f_480p_20260706_194715`
- Description: online adaptive SeaCache using the speedup-conditioned MiniDiT row-split checkpoint, same six prompts/baselines and generation parameters as the 2026-06-30 MiniDiT split-compare run, but sweeping target speedup by PSNR target: target 22 uses `2.2/2.5/2.8`, target 28 uses `1.4/1.7/2.0`.
- Results: completed `36/36`, no failed rows. OpenVid train target 22: speedup targets `2.2/2.5/2.8` produced `2.238x/2.401x/2.675x` and `24.111/22.562/22.504 dB`. OpenVid train target 28: speedup targets `1.4/1.7/2.0` produced `1.398x/1.685x/2.043x` and `31.652/27.405/25.455 dB`. VBench10 target 22: `2.264x/2.352x/2.709x` and `19.146/17.198/17.358 dB`. VBench10 target 28: `1.401x/1.666x/2.067x` and `27.930/23.581/18.652 dB`.
- Report: `reports/report_predictor_speedup.md`, with plot `reports/assets/predictor_speedup_training_loss_curves.svg`.
- Conclusion: target speedup controls threshold/reuse and actual speed, but quality calibration remains dataset-dependent. VBench10 target 28 with target speedup `1.4` is well calibrated (`1.401x`, `27.930 dB`); aggressive speed targets undershoot PSNR sharply on VBench10.

### Predictor Max-Condition Behavior Probe

- Roots: `/hy-tmp/wan22_predictor_max_condition_behavior_20260708`, `/hy-tmp/wan22_predictor_condition_sensitivity_probe_gpu_20260708`
- Description: probed whether speedup-conditioned threshold prediction depends on latent/timestep/PSNR or mostly on `target_speedup`, using completed online MiniDiT traces, a direct speedup-only condition-only intervention, and a GPU raw-latent MiniDiT intervention. Added reusable raw-latent MiniDiT probe script `adaptive_threshold_predictor/probe_condition_sensitivity.py`. In the GPU run, real train/val latents were sampled by source `step_index=0,12,24,36,49` and bound to their source timestep; random matched-normal latents were reported separately as OOD.
- Results: online MiniDiT traces contain `3,600` predictor calls. Categorical mean R2 for predicted threshold was `0.999404` using `target_speedup` alone, `0.999557` with `target_speedup + step_index`, `0.999601` with `target_speedup + sample_id`, and unchanged at `0.999404` with `target_psnr + target_speedup`. In the speedup-only condition-only intervention, fixed `target_speedup=3.5` while varying pseudo latent/timestep/PSNR produced threshold range `0.000215`; fixed `target_psnr=45` while varying speedup produced threshold range `0.689347`. In the GPU raw-latent MiniDiT probe, fixed `target_speedup=3.5` on source-bound real latents produced mean threshold `0.766872` with range `0.118354`; mean threshold decreased from `0.793165` at PSNR `18` to `0.723120` at PSNR `45`. Fixed `target_psnr=45` while varying speedup produced mean thresholds from `0.100410` at speedup `1.1` to `0.723120` at speedup `3.5`, range `0.671022`.
- Report: `reports/report_predictor_max_condition_behavior.md`
- Focused report: `reports/report_max_speedup_psnr.md`
- Conclusion: current speedup-conditioned inverse-task predictors are strongly dominated by `target_speedup`, especially when speedup is varied directly. The actual MiniDiT checkpoint is not literally independent of PSNR: at fixed high `target_speedup=3.5`, higher requested PSNR lowers the predicted threshold. Latent/source-step effects are smaller than the speedup sweep and are reported with source timestep binding to avoid OOD timestep-latent pairs.

### Reports and Readouts

- Cache summary reports are under `reports/`, especially `report_cache_experiments_summary.md`, `report_vbench10_zeus_threshold_seacache.md`, and `report_sampling_solver_impact_zeus_seacache_20260630.md`.
- Adaptive predictor reports include:
  - `reports/report_adaptive_predictor.md`
  - `reports/report_adaptive_predictor_training_curves.md`
  - `reports/report_transformer_predictor_architecture.md`
  - `reports/report_gated_multifeature_mlp_architecture.md`
  - `reports/report_mini_dit_transformer_predictor_comprehensive_20260630.md`
  - `reports/report_gated_multifeature_mlp_predictor_comprehensive_20260630.md`
  - `reports/report_adaptive_predictor_mini_dit_vs_gated_mlp_comparison_20260630.md`
  - `reports/report_predictor_speedup.md`
  - `reports/report_predictor_max_condition_behavior.md`
  - `reports/report_max_speedup_psnr.md`
- Conclusion: reports should be treated as the detailed source of tables and plots; this file is only the short handoff index.

### Path Lookup Notes

- 2026-07-04: located Wan2.2 SeaCache Ali-10 prompt-02 artifacts. Complete Ali-10 UniPC run root is `/hy-tmp/wan22_seacache_unipc_ali10_50step_45f_480p_20260627_023222`; prompt-02 sample id is `ali_002`, with videos under `baseline/ali_002.mp4` and `seacache/th_*/ali_002.mp4`. Earlier DPM++ prompt-02 pilot roots are `/hy-tmp/wan22_seacache_prompt02_dense_20260611_204826` and `/hy-tmp/wan22_seacache_prompt02_highthr_20260612_000218`.
- 2026-07-04: checked DPM++ prompt-02 pilot parameters against the full Ali-10 UniPC prompt-02 run. Prompt text, seed `42`, size `832*480`, frame count `45`, steps `50`, shift `12.0`, guide scale `(3.0, 4.0)`, SeaCache threshold parameters, block cache disabled, and CFG cache disabled all match. The material experimental difference is `sample_solver=dpm++` for the pilot versus `sample_solver=unipc` for the Ali-10 run; DPM++ reused a matching older DPM++ baseline from `/hy-tmp/wan22_zeus_threshold_reuse_interp_10prompt_5th_20260608_195427`.
- 2026-07-08: prepared current adaptive predictor speedup-condition code, reports, logs, and experiment-result symlinks for repository handoff. Session log: `logs/session_20260708_commit_push.md`.
- 2026-07-08: reviewed `adaptive_threshold_predictor/` for handoff clarity, expanded its README with a file map/current workflow/caveats, and fixed misleading script help/output text. Session log: `logs/session_20260708_adaptive_threshold_predictor_handoff_review.md`.
- 2026-07-08: probed predictor behavior near high speedup/high PSNR targets; GPU raw-latent bound-timestep probe shows speedup dominates, while MiniDiT still lowers threshold as requested PSNR increases at fixed high speedup. Session log: `logs/session_20260708_predictor_max_condition_behavior.md`.

## Common Errors And Solutions

### GPU Mode Is Not Enabled

- Symptom: `nvidia-smi` prints `No devices were found`.
- Cause: the cloud instance can boot without GPU attachment.
- Solution: do not launch inference, video generation, PSNR batches, or GPU training. Ask for or switch to GPU mode first, then confirm A100 visibility with `nvidia-smi`.

### Speedup Uses Wrong Time Metric

- Symptom: speedups look inconsistent because model loading, video saving, or process startup time is included.
- Cause: using full wall-clock process time instead of compute-only inference time.
- Solution: use `inference_compute_elapsed_seconds` from logs or summary tables. Only use wall time for operational cost estimates when explicitly stated.

### FFmpeg/ffprobe Not Found In tmux

- Symptom: PSNR or ffprobe steps fail inside tmux even though they work interactively.
- Cause: tmux PATH may not include the conda environment binaries.
- Solution: call FFmpeg/ffprobe from the conda environment path, or make runners search known paths such as `/hy-tmp/miniconda3/envs/Wan2.2/bin/ffmpeg`.

### Single-Process Batch Runner OOM From Cache Object Retention

- Symptom: memory grows candidate by candidate; `torch.cuda.empty_cache()` does not recover enough memory; later candidates OOM.
- Cause: runner/factory keeps references to old SeaCache/adaptive SeaCache/replay cache instances. Those instances hold GPU tensors such as `previous_feature`, `previous_residual`, and current latent snapshots.
- Solution: after each candidate, write summary/trace, call cache `clear_runtime_state()`, call factory `clear_last_instance()`, restore patched classes when needed, delete factory references, and then call `torch.cuda.empty_cache()`. Do not keep historical `instances` lists.

### AdaCache All-Block Residual Cache OOM

- Symptom: local official-style AdaCache candidate OOMs at default `832*480`, `45f`, `50` steps on single A100 80GB.
- Cause: all-block residual caching with separate cond/uncond state exceeds available memory.
- Solution: do not rerun the all-block variant locally at full size. Use selected-block caching, CPU/offloaded cache tensors, smaller shape, or imported reproduction results.

### Full No-Offload Wan2.2 T2V-14B Does Not Fit Single A100 80GB

- Symptom: no-offload experiments run out of memory or leave too little headroom for caches.
- Cause: high/low DiT checkpoints, T5, activations, and cache tensors exceed practical single-GPU memory.
- Solution: keep default `--offload_model` and `--convert_model_dtype` unless using model parallel/FSDP or a smaller test shape.

### Branch Detection By Call Parity Is Wrong For This Project

- Symptom: cache reuse/recompute counts or outputs become inconsistent when combining CFG/timestep/block caches or switching Wan2.2 high/low stages.
- Cause: inferring cond/uncond branch from model-call parity, as in some reference code, breaks under Wan2.2 stage switching and outer CFG cache skipping.
- Solution: cache state must be keyed explicitly by `model_stage` and `branch`.

### Wrong Cache Composition Order

- Symptom: block cache runs even when timestep cache should have reused, or CFG cache does not skip uncond correctly.
- Cause: cache checks are composed in the wrong order.
- Solution: keep the project-standard order: CFG cache outermost; for each actual cond/uncond branch, check timestep cache first; only on timestep miss enter block cache; only when timestep and block miss execute transformer blocks.

### SeaCache Scheduler/Feature Alignment Bugs

- Symptom: local SeaCache behavior differs from official reference in forced recompute windows or threshold decisions.
- Cause: filtering forced ret/cutoff features, using the wrong scheduler sigma index, or resetting accumulated distance on reuse.
- Solution: match official behavior: metric feature is first block modulated norm input; middle-step filtering uses `scheduler.sigmas[idx]`; forced ret/cutoff/history-missing recompute stores the unfiltered feature; accumulated relative L1 resets only on threshold-crossing recompute or forced recompute windows; `previous_feature` updates on every call.

### Sea-Style Block Cache Memory Pressure

- Symptom: high-threshold sea block-group grids risk OOM.
- Cause: `sea_full_rel_l1` stores full filtered features per block group in addition to cached residuals.
- Solution: start with small pilots, conservative thresholds, or fewer combinations; inspect `failed/` and resume with existing artifacts if a long grid fails.

### CFG Cache Miss Handling Can Skew Comparisons

- Symptom: CFG cache experiments compare different behaviors unintentionally.
- Cause: toggling forced uncond recomputation on miss or accumulating skipped uncond branch state inconsistently.
- Solution: for current threshold-combination runs, leave `cfg_force_uncond_recompute_on_miss` disabled unless explicitly doing an ablation. For sea timestep + sea CFG, keep No-Skip-Accum behavior for skipped uncond branches.

### Reuse/Recompute Counts Are Misread

- Symptom: summary counts appear doubled or inconsistent with timestep count.
- Cause: branch-call counts are summed across cond/uncond and high/low stages, while unique timestep counts are different.
- Solution: distinguish unique timesteps from summed branch-call counts in reports and tables.

### PSNR Contains Infinity/Perfect Frames

- Symptom: PSNR aggregate is infinite or misleading.
- Cause: FFmpeg reports perfect/Infinity frames when frames match exactly.
- Solution: for summary PSNR, exclude perfect/Infinity frames where applicable and record both PSNR JSON/log paths in archives.

### Baseline Mismatch

- Symptom: PSNR comparison is invalid or unexpectedly low.
- Cause: candidate is compared to a baseline with different prompt, seed, size, frame count, solver, or sampler.
- Solution: baseline must match prompt/sample ID, seed `42`, size `832*480`, `45` frames, `50` steps, solver, and dtype/offload setting. DPM++ baselines cannot be reused as UniPC PSNR references.

### Dataset Split Leakage

- Symptom: offline validation MAE is very low but online generalization remains poor.
- Cause: row split shares source-video identities between train and validation; it measures same-video interpolation, not held-out-video generalization.
- Solution: use sample split as the primary offline generalization signal. Treat row split as a diagnostic for capacity and within-video fitting only.

### Adaptive Predictor Offline Task Does Not Match Online Control

- Symptom: predictor has reasonable offline MAE but misses online target PSNR.
- Cause: `candidate_inverse` trains on fixed-threshold candidate latents and achieved PSNR, while online inference uses adaptive-run latents and desired PSNR.
- Solution: do not select models by offline MAE alone. Validate online against fixed-threshold SeaCache on the same prompts, and consider redesigning labels/training to match target-control inference.

### Speedup-Conditioned Predictor Checkpoint Incompatibility

- Symptom: online adaptive SeaCache fails to load older predictor checkpoints after adding `target_speedup`.
- Cause: older two-condition checkpoints have condition-layer weights shaped for `(timestep, target_psnr)`, for example `cond_embed.0.weight` with second dimension `2`; current models instantiate three-condition layers for `(timestep, target_psnr, target_speedup)`.
- Solution: retrain predictors with the speedup-conditioned pipeline, or explicitly reject old checkpoints before launching long GPU runs. Do not use old June two-condition checkpoints as defaults for new speedup-conditioned experiments.

### Adaptive Runner Target Speedup Metadata

- Symptom: adaptive experiment results can become ambiguous when comparing multiple desired speedups.
- Cause: older runners initially patched in a compatibility `target_speedup=2.0` constant.
- Solution: adaptive runners now expose `--target_speedup` with compatibility default `2.0` and archive it in target env files, config JSON, command records, failed records, summary CSV rows, and MiniDiT aggregate rows. If doing a full target-speedup grid, extend the runner loop explicitly rather than overloading this single-value option.

### Range Mismatch In Threshold Predictor Outputs

- Symptom: MLP predictor emits thresholds outside the intended SeaCache range or differs from MiniDiT output semantics.
- Cause: direct sigmoid output `[0,1]` instead of scaled threshold range.
- Solution: use `threshold = min_threshold + sigmoid(raw) * (max_threshold - min_threshold)` with current defaults `[0.10,0.80]`.

### OpenVid Archives Can Fill `/hy-tmp`

- Symptom: extraction/download leaves too little free disk for experiments.
- Cause: compressed OpenVid prompt archives plus extracted trace data are large; extracted trace root is about `135G`.
- Solution: check `/hy-tmp` free space before downloading or extracting. Prefer symlink-based flat layouts rather than copying artifacts.

### Do Not Store Large Artifacts In Git/System Disk

- Symptom: repository or system disk grows unexpectedly.
- Cause: copying videos, model weights, checkpoints, or caches into tracked paths or `/root`.
- Solution: keep large artifacts under `/hy-tmp` and use `experiment_results/` symlinks for discoverability. Track only code, small reports, small tables, and handoff logs.

### VS Code Remote Extension Host Killed

- Symptom: VS Code Remote repeatedly disconnects with extension host `SIGKILL`.
- Cause: indexing large experiment/model/cache symlinks overwhelms the remote extension host.
- Solution: keep `.vscode/settings.json` exclusions for large directories and disable expensive Python/Pylance indexing/test discovery in this workspace.

### Missing Full SeaCache DPM++ Ali-10 Result

- Symptom: someone tries to compare full SeaCache DPM++ versus UniPC on Ali-10.
- Cause: only DPM++ prompt-01/02 pilots exist; no complete Ali-10 DPM++ SeaCache root was found.
- Solution: state that full Ali-10 sampler comparison is unavailable, or run a formal full Ali-10 DPM++ SeaCache experiment before reporting that comparison.

### Stale tmux/Progress Notes

- Symptom: `PROGRESS.md` says a run is active even though it has completed.
- Cause: old launch/progress entries were not replaced by completion summaries.
- Solution: this file should not store launch-state chatter. Record only final result/conclusion, and use logs or runner roots for detailed chronology.
