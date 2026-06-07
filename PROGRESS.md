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

## Notes

- Follow `AGENTS.md` workflow: read this file at session start, update it before session end, and keep concise session logs under `logs/`.
- Experiment outputs, model weights, caches, logs, and result tables should stay under `/hy-tmp`.
