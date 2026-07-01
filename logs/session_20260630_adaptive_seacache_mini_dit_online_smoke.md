# Session 2026-06-30 Adaptive SeaCache MiniDiT Online Smoke

## What Changed

- Added MiniDiT/Transformer predictor support to `adaptive_seacache_wan22/cache.py`.
  - New `AdaptiveSeaCacheGateConfig.model_type` supports `auto`, `mlp`, and `mini_dit_cls`.
  - `OnlineAdaptiveThresholdGate` preserves the legacy cached-feature MLP path.
  - `mini_dit_cls` checkpoints instantiate `MiniDiTCLSAdaptiveThresholdPredictor` and feed the raw current Wan latent directly to the model.
  - Checkpoint loading supports `best_model_checkpoint.pt` payloads with `model_state_dict`, `args`, and `feature_extractor`; plain state dicts still work with adjacent `config.json` where available.
- Added MiniDiT CLI options to `adaptive_seacache_wan22/generate_t2v.py`.
  - `--adaptive_model_type auto|mlp|mini_dit_cls`
  - optional `--adaptive_dit_input_shape`, `--adaptive_dit_patch_size`, `--adaptive_dit_dim`, `--adaptive_dit_layers`, `--adaptive_dit_heads`, `--adaptive_dit_mlp_ratio`, `--adaptive_dit_dropout`, `--adaptive_dit_gate_init`
  - Clears stale logging handlers before calling the Wan generator so `Timestep cache summary` and compute timing are emitted.
- Updated `adaptive_seacache_wan22/README.md` with MiniDiT usage and the first online smoke result.
- Appended this work to `PROGRESS.md`.

## Validation

- Ran `py_compile` on:
  - `adaptive_seacache_wan22/cache.py`
  - `adaptive_seacache_wan22/generate_t2v.py`
- Loaded the trained MiniDiT checkpoint and ran dummy latent predictions:
  - checkpoint: `/hy-tmp/wan22_adaptive_threshold_mini_dit_cls_convpatch_rowsplit_packed_d96_l2_bs128_20260629_232659/best_model_checkpoint.pt`
  - auto-detected `model_type=mini_dit_cls`
  - input shape `(16, 12, 60, 104)`
  - patch size `(3, 12, 8)`
  - threshold bounds `0.1-0.8`
- Confirmed GPU mode before full inference: A100 80GB was available and idle.

## Full Smoke Run

- Result root: `/hy-tmp/wan22_adaptive_seacache_mini_dit_vbench01_smoke_20260630_021304`
- Prompt: `A woman is playing football.`
- Dataset/sample: `vbench10_001`
- Seed: `42`
- Size: `832*480`
- Frames: `45`
- Steps: `50`
- Solver: `dpm++`
- Cache mode: timestep-only adaptive SeaCache
- Target PSNR: `25`
- Predictor checkpoint: `/hy-tmp/wan22_adaptive_threshold_mini_dit_cls_convpatch_rowsplit_packed_d96_l2_bs128_20260629_232659/best_model_checkpoint.pt`

Outputs:

- Video: `/hy-tmp/wan22_adaptive_seacache_mini_dit_vbench01_smoke_20260630_021304/videos/vbench01_target25.mp4`
- Run log: `/hy-tmp/wan22_adaptive_seacache_mini_dit_vbench01_smoke_20260630_021304/logs/run.log`
- ffprobe: `/hy-tmp/wan22_adaptive_seacache_mini_dit_vbench01_smoke_20260630_021304/ffprobe/vbench01_target25.json`
- PSNR log: `/hy-tmp/wan22_adaptive_seacache_mini_dit_vbench01_smoke_20260630_021304/psnr/vbench01_target25_psnr.log`
- Summary table: `/hy-tmp/wan22_adaptive_seacache_mini_dit_vbench01_smoke_20260630_021304/results/summary.csv`
- Per-key cache summary: `/hy-tmp/wan22_adaptive_seacache_mini_dit_vbench01_smoke_20260630_021304/results/cache_key_summary.csv`

Metrics:

- ffprobe: `832x480`, `45` frames, `16 fps`, duration `2.8125s`
- Compute elapsed: `374.941s`
- Wall elapsed: `436.478s`
- Matching dpm++ no-cache baseline compute elapsed: `538.211s`
- Speedup vs baseline compute: `1.435x`
- FFmpeg PSNR average: `19.493 dB`
- FFmpeg PSNR min/max: `16.647 / 21.979 dB`
- Total reuse branch calls: `42`
- Total recompute branch calls: `58`
- Predicted threshold range: `0.1885-0.2852`
- Mean predicted threshold across cache keys: about `0.2068`

## Notes

- The online MiniDiT path works end to end and produces adaptive threshold paths in the Wan timestep cache summary.
- The first target-25 smoke significantly undershot target quality on this prompt. The next useful experiment is a small target sweep or replay comparison against fixed SeaCache thresholds on the same prompt before scaling to VBench10.
- An earlier run at `/hy-tmp/wan22_adaptive_seacache_mini_dit_vbench01_smoke_20260630_020009` generated successfully but had incomplete Wan info logs because logging handlers were already configured. It was superseded by the `20260630_021304` run after fixing logging.
