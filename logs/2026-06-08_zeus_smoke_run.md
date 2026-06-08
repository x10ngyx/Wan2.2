# 2026-06-08 ZEUS smoke run

- Read `PROGRESS.md`, ZEUS timestep cache code in `generate.py`, `wan/text2video.py`, `wan/timestep_cache.py`, and the ZEUS experiment scripts under `experiments/zeus_timestep_cache_50step_45f_480p/`.
- Confirmed the current instance is GPU-enabled with NVIDIA A100 80GB PCIe; the `Wan2.2` conda env sees CUDA with PyTorch `2.4.0+cu121`.
- Fixed environment blockers for running `generate.py`: installed missing `decord`, S2V/Animate import dependencies such as `librosa`, `onnxruntime`, `peft`, `sentencepiece`, `lightning`, `modelscope`, `GitPython`, and installed conda-forge `ffmpeg`/`ffprobe`.
- Validation after dependency install: `generate.py --help` succeeds, `ffprobe -version` succeeds, `wan` imports successfully. `pip check` still reports `decord 0.6.0 is not supported on this platform`; this did not block import or T2V inference.
- Ran smoke experiment root `/hy-tmp/wan22_smoke_zeus_20260608_0951`:
  - Baseline: `t2v-A14B`, prompt 01, seed `20260608`, `832*480`, `frame_num=5`, `sample_steps=8`, `sample_solver=dpm++`, `offload_model=True`, `convert_model_dtype`, `timestep_cache=none`.
  - ZEUS: same prompt/seed/shape, `timestep_cache=zeus`, `zeus_acc_start=3`, `zeus_acc_end=8`, `zeus_denominator=2`, `zeus_modular=(1,)`, `zeus_caching_mode=reuse_interp`, `zeus_lagrange_term=0`.
- Outputs:
  - Baseline video: `/hy-tmp/wan22_smoke_zeus_20260608_0951/baseline/prompt_01.mp4`
  - ZEUS video: `/hy-tmp/wan22_smoke_zeus_20260608_0951/zeus/prompt_01.mp4`
  - Summary CSV: `/hy-tmp/wan22_smoke_zeus_20260608_0951/results/summary.csv`
- ffprobe validation for both videos: `832x480`, `5` frames, `16 fps`, duration `0.312500s`.
- ZEUS cache summary from raw log: high/cond reuse `2`, high/uncond reuse `2`, low/cond reuse `0`, low/uncond reuse `0`; total reuse `4`, recompute `12`.
- Smoke summary: baseline elapsed `375s`, ZEUS elapsed `374s`, speedup `1.0027x`, mean PSNR `14.9169`, min PSNR `13.6523`, max PSNR `15.8253`. This was only a pipeline smoke test with forced early skipping, not a formal quality/speed setting.
