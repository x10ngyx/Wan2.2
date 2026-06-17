# 2026-06-16 TaylorSeer Third-Party Move And Multi-GPU Prep

## Summary

- Moved the standalone TaylorSeer Wan2.2 integration into `third_party/taylorseer_wan22/`.
- Main Wan2.2 cache code remains untouched; `generate.py`, `wan/text2video.py`, `wan/timestep_cache.py`, and `wan/modules/model.py` contain no TaylorSeer integration.
- Added multi-GPU launch support to the third-party runner for future experiments on a multi-GPU machine.

## Files

- Added `third_party/__init__.py`.
- Moved/updated:
  - `third_party/taylorseer_wan22/cache.py`
  - `third_party/taylorseer_wan22/patch.py`
  - `third_party/taylorseer_wan22/text2video.py`
  - `third_party/taylorseer_wan22/generate_t2v.py`
  - `third_party/taylorseer_wan22/README.md`

## Official Logic Check

- The implementation follows the public TaylorSeer-Wan2.1 block-level method:
  - `cond_stream` decides whether the current step is `full` or `Taylor`.
  - `uncond_stream` follows the same step type.
  - cond/uncond module caches are separate.
  - full steps cache each block's `self-attention`, `cross-attention`, and `ffn` outputs.
  - Taylor steps reconstruct those module outputs with cached derivative dictionaries and the Taylor formula.
- Wan2.2-specific adaptation:
  - `high_noise_model` and `low_noise_model` have separate TaylorSeer cache states.
  - No cache is shared across the high/low denoiser boundary.

## Multi-GPU Support

- Runner supports:
  - `torchrun`
  - `--ulysses_size <world_size>`
  - `--dit_fsdp`
  - `--t5_fsdp`
- Patch code now unwraps FSDP modules and patches the underlying model blocks.
- With Ulysses sequence parallel, Wan2.2 shards tokens before block execution; TaylorSeer caches per-rank local module-output shards.
- Runtime multi-GPU validation is pending because this machine currently has no visible GPU.

## Validation

- Passed:
  - `/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m py_compile third_party/__init__.py third_party/taylorseer_wan22/__init__.py third_party/taylorseer_wan22/cache.py third_party/taylorseer_wan22/patch.py third_party/taylorseer_wan22/text2video.py third_party/taylorseer_wan22/generate_t2v.py`
  - `/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m third_party.taylorseer_wan22.generate_t2v --help`
  - `rg -n "TaylorSeer|taylorseer" generate.py wan/text2video.py wan/timestep_cache.py wan/modules/model.py`
