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
- Removed incomplete local model artifacts `/hy-tmp/models/Wan2.2-T2V-A14B`, `/hy-tmp/models/Wan2.2-T2V-A14B.tar`, and `/hy-tmp/Wan2.2-T2V-A14B.tar` before redownloading.
- Logged into HyCloud OSS with the user-provided account and downloaded `oss://datasets/Wan2.2-T2V-A14B.tar` to `/hy-tmp/models/Wan2.2-T2V-A14B.tar`; OSS reported successful download of `117.53GB` in about `30m21s`.
- Extracted the archive into `/hy-tmp/models/Wan2.2-T2V-A14B`. Verification found `31` extracted files matching the archive manifest, including 6 high-noise and 6 low-noise safetensor shards, `Wan2.1_VAE.pth`, and `models_t5_umt5-xxl-enc-bf16.pth`.
- Post-extraction disk usage: `/hy-tmp/models/Wan2.2-T2V-A14B` is `118G`, archive is `118G`, and `/hy-tmp` has about `154G` available.
- Added `zeus-threshold` timestep cache as a separate implementation without modifying the original `ZeusTimestepCache` classes. The new path compares current input latent relative-L1 distance against `--zeus_threshold` to decide reuse vs recompute, while preserving the existing `(model_stage, branch)` state separation and ZEUS output reuse modes.
- Added CLI support for `--timestep_cache zeus-threshold` with `--zeus_threshold`, `--zeus_threshold_metric rel_l1`, and `--zeus_threshold_eps`. Block cache and CFG cache remain unimplemented placeholders.
- Validation for `zeus-threshold`: conda and system `py_compile` passed for `generate.py`, `wan/text2video.py`, and `wan/timestep_cache.py`; a CPU dummy import-by-path check verified original fixed ZEUS skip behavior and threshold low/high latent-distance decisions.
- Confirmed the original ZEUS implementation remains unchanged in `wan/timestep_cache.py`; the diff only appends new threshold classes. A corrected CPU smoke test verified original fixed-policy ZEUS skip/recompute and `reuse_interp` behavior.
- Added ZEUS-threshold experiment scripts under `experiments/zeus_threshold_50step_45f_480p/` for Wan2.2 T2V-A14B, 50 steps, 45 frames, 480p, 10 prompts from `prompt.txt`, baseline plus five default thresholds `0.03 0.08 0.15 0.30 0.60`.
- The threshold experiment script archives generated videos, full command scripts, raw logs, timing files, ffprobe JSON, PSNR JSON/logs vs baseline, failure records, per-prompt summary CSV, and aggregate-by-threshold CSV/JSON under `/hy-tmp`.
- Validation for the threshold experiment scripts: `bash -n` passed for `run_experiments.sh`, conda `py_compile` passed for the summarizer and reused helper scripts, and the prompt parser returned 10 prompts. Full generation was not run because the current instance still has no visible GPU.
- Current session re-read AGENTS workflow, `PROGRESS.md`, README, main T2V generation path, timestep cache implementation, and experiment scripts. Confirmed current HEAD includes ZEUS timestep cache and ZEUS-threshold cache, while block cache and CFG cache are still placeholder CLI interfaces only.
- Current working tree before this session already had untracked `doc/` PDFs, untracked OSS/download logs, and a modified `PROGRESS.md`; this session only added this progress note and a concise session log.
- Reviewed third-party experiment PDFs under `doc/`. Key takeaways: ZEUS-style timestep skipping has a useful quality/speed tradeoff, especially later/mid-late timestep ranges; naive specific-block skipping produced poor PSNR/SSIM despite modest speedups; step-level residual analysis recommends prioritizing output relative-L2 over cosine similarity, with mid steps most cache-friendly.
- Installed lightweight `pypdf` in the existing Wan2.2 conda environment to extract text from the local PDF reports.

- Current GPU-mode instance is usable: `nvidia-smi` reports NVIDIA A100 80GB PCIe, and `/hy-tmp/miniconda3/envs/Wan2.2/bin/python` reports `torch.cuda.is_available() == True`.
- Completed environment dependency cleanup needed for `generate.py` import and video validation. Installed missing optional import dependencies including `decord`, S2V/Animate-related PyPI packages such as `librosa`, `onnxruntime`, `peft`, `sentencepiece`, `lightning`, `modelscope`, `GitPython`, and installed conda-forge `ffmpeg`/`ffprobe` into the `Wan2.2` env.
- Post-install validation: `generate.py --help` succeeds; `ffprobe` reports version `8.1.1`; PyTorch remains `2.4.0+cu121` with CUDA visible. `pip check` still reports `decord 0.6.0 is not supported on this platform`, but `wan` imports successfully and T2V generation ran successfully.
- Ran a small ZEUS smoke experiment at `/hy-tmp/wan22_smoke_zeus_20260608_0951` using prompt 01, seed `20260608`, `t2v-A14B`, `832*480`, `frame_num=5`, `sample_steps=8`, `sample_solver=dpm++`, `offload_model=True`, and `convert_model_dtype`.
- Smoke baseline and ZEUS videos both generated successfully and were validated with `ffprobe`: `832x480`, `5` frames, `16 fps`, duration `0.312500s`. Command scripts, raw logs, timing files, ffprobe JSON, PSNR JSON, summary CSV, and aggregate JSON are archived under the smoke experiment root.
- Smoke ZEUS run used `--zeus_acc_start 3 --zeus_acc_end 8 --zeus_denominator 2 --zeus_modular '(1,)' --zeus_lagrange_term 0` to force cache hits in a short run. Cache summary: high/cond reuse `2`, high/uncond reuse `2`, low/cond reuse `0`, low/uncond reuse `0`; total reuse `4`, recompute `12`.
- Smoke summary: baseline elapsed `375s`, ZEUS elapsed `374s`, speedup `1.0027x`, mean PSNR `14.9169` over `5` frames. This is only a pipeline smoke test; the tiny step/frame count and forced early skipping are not a quality/speed setting for formal evaluation.


- Updated ZEUS experiment requirements to align with `doc/wan22_zeus_test - 视频质量与加速比评估报告.pdf`: use fixed seed `42` for all prompts, use FFmpeg `psnr` filter `psnr_avg` with perfect frames (`PSNR > 100 dB`) excluded, and report speedup from compute-only inference time rather than full process wall time.
- Updated `experiments/zeus_timestep_cache_50step_45f_480p/compute_psnr.py` to use the document-aligned FFmpeg PSNR method. The previous OpenCV decoded-frame PSNR for prompt 01 was backed up as `prompt_01.opencv_psnr.json` in the stopped run; FFmpeg PSNR for that prompt changed from `22.0652` to `25.0556`.
- Updated ZEUS timestep and threshold experiment scripts to default `BASE_SEED=42` and use fixed seed `42` for every prompt instead of `BASE_SEED + i`.
- Updated T2V timing instrumentation: `wan/text2video.py` logs `inference_compute_elapsed_seconds` and `inference_weight_transfer_elapsed_seconds`; experiment `.time` files now prefer compute-only inference time, excluding model loading, video saving, T5/high/low weight transfers, high/low CPU/GPU switching, and final offload. T5 encoding, denoising compute, scheduler steps, and VAE decode remain included.
- Confirmed true no-offload is not appropriate on the current A100 80GB setup without model-parallel changes: high and low noise model directories are about `54G` each, plus `11G` T5 and activations. Current `offload_model=False` would also not fully disable stage movement because `WanT2V(init_on_cpu=True)` remains the default.
- Created workspace result symlink directory `experiment_results/` and linked all `/hy-tmp/wan22_*` experiment result roots there for easier inspection from the repo.
- Restarted the formal ZEUS timestep experiment in tmux after applying the aligned requirements. Current tmux session: `zeus_timestep_full_114307`; result root: `/hy-tmp/wan22_zeus_timestep_cache_50step_45f_480p_full_20260608_114307`; workspace symlink: `experiment_results/wan22_zeus_timestep_cache_50step_45f_480p_full_20260608_114307`.
- 2026-06-08 12:18 CST monitor check: tmux session zeus_timestep_full_114307 is still running. Prompt 01 completed for baseline and ZEUS with validated videos, FFmpeg PSNR output, interim summary, and aggregate JSON archived. Prompt 01 compute-only timing: baseline 522.872s, ZEUS 263.628s, speedup 1.983x; FFmpeg mean PSNR 22.226 dB; total ZEUS reuse/recompute 50 / 50. Prompt 02 baseline is currently running and was at sampling step 32/50 in the tmux pane.
- 2026-06-08 16:03 CST completion check: formal ZEUS timestep experiment finished and tmux is no longer running. Result root /hy-tmp/wan22_zeus_timestep_cache_50step_45f_480p_full_20260608_114307 contains 10 baseline videos, 10 ZEUS videos, command scripts, raw logs, timing files, ffprobe JSON, PSNR JSON/logs for all prompts, final summary.csv, and aggregate.json. No failure files were found. Aggregate results: num_pairs 10, total baseline compute time 5262.025s, total ZEUS compute time 2649.240s, overall speedup 1.986x, mean FFmpeg PSNR 23.705 dB.- 2026-06-08 reuse/recompute summary fix: found that zeus_reuse_count and zeus_recompute_count in experiment summaries were summing branch-level states across high/low and cond/uncond, which doubled the intended timestep-level counts. Updated both ZEUS timestep and ZEUS-threshold summarizers so the existing count columns report unique timestep counts from skipping_path/recompute_path, while new branch-call columns preserve summed cond/uncond model-call counts. Regenerated the completed formal ZEUS summary; prompt-level counts are now 25 reuse / 25 recompute timesteps, with 50 / 50 branch calls retained. The pre-fix CSV was backed up as summary.branch_count_before_fix.csv.

## Notes

- Follow `AGENTS.md` workflow: read this file at session start, update it before session end, and keep concise session logs under `logs/`.
- Experiment outputs, model weights, caches, logs, and result tables should stay under `/hy-tmp`.
