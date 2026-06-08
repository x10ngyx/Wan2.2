# 2026-06-08 ZEUS experiment requirements update

This session refined the formal ZEUS timestep experiment requirements after inspecting `doc/wan22_zeus_test - 视频质量与加速比评估报告.pdf` and the first prompt output from the initial run.

## Required experiment alignment

- Use fixed seed `42`, matching the PDF report. Do not use `BASE_SEED + i`; every prompt should use seed `42` for document-aligned testing.
- Use `832*480`, `45` frames, `50` denoising steps, and `dpm++` from the formal ZEUS timestep experiment script.
- Use FFmpeg PSNR, not OpenCV decoded-frame MSE PSNR:
  - FFmpeg `psnr` filter
  - parse per-frame `psnr_avg`
  - use YUV weighted average as produced by FFmpeg
  - exclude perfect frames where `PSNR > 100 dB`
- Record speedup from compute-only inference time, not full process wall time.

## Timing definition

The `.time` files should use `inference_compute_elapsed_seconds` when present.

Included in compute-only inference time:
- T5 text encoding compute
- denoising step compute, including model forward or cache reuse and scheduler step
- VAE decode

Excluded from compute-only inference time:
- model loading and pipeline construction
- video file saving
- T5 weight transfer to/from GPU
- high/low noise model CPU/GPU transfer in `_prepare_model_for_timestep`
- final high/low offload after sampling

`wan/text2video.py` also logs `inference_weight_transfer_elapsed_seconds` for visibility. `generate.py` logs the broader post-load wall time as `generation_wall_elapsed_seconds`, but experiment scripts do not use it for speedup.

## Offload decision

True no-offload is not appropriate for this A100 80GB setup without model-parallel changes. The high and low noise model directories are about `54G` each, and T5 is about `11G`, before activations. Also, the current native `WanT2V` path defaults to `init_on_cpu=True`, so simply passing `--offload_model False` would not make both high/low models resident on GPU.

## Code/script changes made

- `experiments/zeus_timestep_cache_50step_45f_480p/compute_psnr.py` now uses the FFmpeg PSNR method above.
- `experiments/zeus_timestep_cache_50step_45f_480p/run_experiments.sh` defaults to `BASE_SEED=42`, uses fixed seed `42` for all prompts, and writes `.time` from `inference_compute_elapsed_seconds`.
- `experiments/zeus_threshold_50step_45f_480p/run_experiments.sh` was updated with the same seed and timing conventions.
- `wan/text2video.py` now emits compute-only inference timing and weight-transfer timing.
- `generate.py` renamed the broader post-load timer to `generation_wall_elapsed_seconds` to avoid confusion.

## Current run

Stopped the earlier run rooted at `/hy-tmp/wan22_zeus_timestep_cache_50step_45f_480p_full_20260608_104311` after prompt 01 inspection. Its OpenCV PSNR result for prompt 01 was preserved as `psnr/prompt_01.opencv_psnr.json`; FFmpeg PSNR recomputation gave mean PSNR `25.0556`.

Restarted the formal run with aligned requirements:
- tmux session: `zeus_timestep_full_114307`
- result root: `/hy-tmp/wan22_zeus_timestep_cache_50step_45f_480p_full_20260608_114307`
- repo symlink: `experiment_results/wan22_zeus_timestep_cache_50step_45f_480p_full_20260608_114307`
