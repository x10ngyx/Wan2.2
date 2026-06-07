# Progress

## 2026-06-06

- Initialized project progress tracking file.
- Current task context: Wan2.2 T2V-14B inference acceleration experiments for timestep cache, block cache, and cfg cache.
- No experiment runs have been recorded in this file yet.

## 2026-06-07

- Installed Miniconda under /hy-tmp/miniconda3 and created conda environment Wan2.2 at /hy-tmp/miniconda3/envs/Wan2.2.
- Configured conda package cache under /hy-tmp/conda-pkgs; pip installs used /hy-tmp/pip-cache and Tsinghua PyPI mirror where practical.
- Installed conservative PyTorch stack: torch==2.4.0+cu121, torchvision==0.19.0+cu121, torchaudio==2.4.0+cu121.
- Installed README dependencies including opencv-python==4.11.0.86, diffusers==0.38.0, transformers==4.51.3, accelerate==1.13.0, numpy==1.26.4, dashscope==1.25.21, imageio-ffmpeg==0.6.0, and flash_attn==2.6.3.
- pip check reported no broken requirements.
- Current instance is not in GPU mode: nvidia-smi reports No devices were found, and torch.cuda.is_available() is False.
- generate.py --help cannot run on this no-GPU instance because import-time code calls torch.cuda.current_device().
- Reviewed ZEUS repository `Ting-Justin-Jiang/ZEUS` at commit `ceff240` and project page for "ZEUS: Zero-shot Efficient Unified Sparsity for Generative Models".
- ZEUS paper PDF/arXiv was not publicly linked on the project page at review time; the project page marks Paper as "Coming Soon".
- Main ZEUS implementation uses Diffusers monkey patches: `zeus/patch.py` attaches `CacheBus`, patches `pipeline.transformer`/`pipeline.unet`, and patches schedulers; `zeus/model.py` bypasses whole denoiser/transformer forward calls on scheduled skip steps; `zeus/solver.py` chooses skip steps and reconstructs skipped outputs via second-order/lagrange interpolation.
- ZEUS Wan support targets Diffusers `WanPipeline` / `WanTransformer3DModel` with `FlowMatchEulerDiscreteScheduler`, demonstrated for `Wan-AI/Wan2.1-T2V-14B-Diffusers`; direct integration into this repo's native Wan2.2 inference path would require porting the scheduler/step-skip logic rather than reusing the patch as-is.
- Implemented ZEUS-style timestep cache for native T2V inference only.
- Added `wan/timestep_cache.py` with ZEUS config/state/helper classes. State is separated by `(model_stage, branch)`, giving independent high/low and cond/uncond histories.
- Wired `WanT2V.generate()` to use the timestep cache around cond/uncond model calls while keeping the scheduler and model classes unchanged.
- Added CLI interface: `--timestep_cache {none,zeus}` plus ZEUS parameters; added placeholder `--block_cache none` and `--cfg_cache none` interfaces without implementing those caches.
- Verified syntax with both system Python and `/hy-tmp/miniconda3/envs/Wan2.2/bin/python` via `py_compile`. Full generation validation still requires a GPU-mode instance.
- Refined the ZEUS timestep cache history gate to require the official three-slot `prev_f` warmup condition before skip eligibility.
- Added experiment scripts under `experiments/zeus_timestep_cache_50step_45f_480p/` for Wan2.2 T2V-A14B, 50 steps, 45 frames, 480p, 10 prompts from `prompt.txt`, baseline vs ZEUS, ffprobe validation, PSNR, timing, speedup, cache reuse/recompute summaries, and failure records.
- Current instance still reports no visible GPU via `nvidia-smi`; full inference/PSNR experiment execution was prepared but not run.

## 2026-06-08

- Verified the HyCloud OSS CLI is installed at `/usr/local/bin/oss`.
- `oss version` reports `v1.2.36+prod`, commit `0759e529f692a8dbca86be24aece46c00abc577d`.
- Confirmed relevant commands for later model-weight downloads: `oss login`, `oss ls -s -d oss://datasets/`, and `oss cp oss://... /hy-tmp/...`; recursive prefix download uses `oss cp oss://bucket/prefix /hy-tmp/path -r`.
- Re-read current in-repo ZEUS implementation logic. Confirmed the implemented path is native T2V timestep cache only: `generate.py` builds `ZeusTimestepCacheConfig`, `WanT2V.generate()` wraps separate cond/uncond model calls, and `wan/timestep_cache.py` maintains independent skip/recompute history per `(model_stage, branch)`.
- Confirmed `--block_cache none` and `--cfg_cache none` remain placeholder CLI interfaces; block cache and CFG cache behavior are not implemented in the current repo.
- Added `zeus-threshold` timestep cache as a separate implementation without modifying the original `ZeusTimestepCache` classes. The new path compares current input latent relative-L1 distance against `--zeus_threshold` to decide reuse vs recompute, while preserving the existing `(model_stage, branch)` state separation and ZEUS output reuse modes.
- Added CLI support for `--timestep_cache zeus-threshold` with `--zeus_threshold`, `--zeus_threshold_metric rel_l1`, and `--zeus_threshold_eps`. Block cache and CFG cache remain unimplemented placeholders.
- Validation for `zeus-threshold`: conda and system `py_compile` passed for `generate.py`, `wan/text2video.py`, and `wan/timestep_cache.py`; a CPU dummy import-by-path check verified original fixed ZEUS skip behavior and threshold low/high latent-distance decisions.
- Confirmed the original ZEUS implementation remains unchanged in `wan/timestep_cache.py`; the diff only appends new threshold classes. A corrected CPU smoke test verified original fixed-policy ZEUS skip/recompute and `reuse_interp` behavior.
- Added ZEUS-threshold experiment scripts under `experiments/zeus_threshold_50step_45f_480p/` for Wan2.2 T2V-A14B, 50 steps, 45 frames, 480p, 10 prompts from `prompt.txt`, baseline plus five default thresholds `0.03 0.08 0.15 0.30 0.60`.
- The threshold experiment script archives generated videos, full command scripts, raw logs, timing files, ffprobe JSON, PSNR JSON/logs vs baseline, failure records, per-prompt summary CSV, and aggregate-by-threshold CSV/JSON under `/hy-tmp`.
- Validation for the threshold experiment scripts: `bash -n` passed for `run_experiments.sh`, conda `py_compile` passed for the summarizer and reused helper scripts, and the prompt parser returned 10 prompts. Full generation was not run because the current instance still has no visible GPU.

## Notes

- Follow `AGENTS.md` workflow: read this file at session start, update it before session end, and keep concise session logs under `logs/`.
- Experiment outputs, model weights, caches, logs, and result tables should stay under `/hy-tmp`.
