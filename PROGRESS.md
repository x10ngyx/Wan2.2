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

## Notes

- Follow `AGENTS.md` workflow: read this file at session start, update it before session end, and keep concise session logs under `logs/`.
- Experiment outputs, model weights, caches, logs, and result tables should stay under `/hy-tmp`.
