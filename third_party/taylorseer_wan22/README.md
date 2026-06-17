# TaylorSeer Wan2.2 Integration

This directory contains a standalone TaylorSeer integration for Wan2.2 T2V-A14B.
It intentionally lives outside the main `wan/` and `generate.py` paths so the
existing ZEUS, SeaCache, block-cache, and CFG-cache experiments are not affected.

## Official Logic Alignment

The implementation follows the public TaylorSeer-Wan2.1 logic from
`Shenyi-Z/TaylorSeer`:

- A step scheduler chooses `full` or `Taylor` using:
  - `fresh_threshold`
  - `max_order`
  - `first_enhance`
  - `cache_counter`
  - `activated_steps`
- `cond_stream` determines the step type first.
- `uncond_stream` follows the same step type for the same denoising step.
- `cond_stream` and `uncond_stream` keep separate module caches.
- On `full` steps, each transformer block records:
  - `self-attention`
  - `cross-attention`
  - `ffn`
- On `Taylor` steps, these three module outputs are reconstructed with the
  Taylor formula from cached derivative dictionaries and are added back through
  the original residual/gating structure.

Wan2.2 differs from the official Wan2.1 code because T2V-A14B has separate
`high_noise_model` and `low_noise_model` denoisers. This implementation keeps
independent TaylorSeer states for:

- `high`
- `low`

No TaylorSeer cache is shared across the high/low model boundary.

## Multi-GPU Support

The runner supports the same broad multi-GPU structure used by the main Wan2.2
runner:

- `torchrun`
- `--ulysses_size <world_size>` sequence parallelism
- `--dit_fsdp`
- `--t5_fsdp`

The TaylorSeer block patch is installed after `WanT2V` construction and supports
normal models and FSDP-wrapped models by patching the underlying module.

When `--ulysses_size > 1`, Wan2.2's sequence-parallel model forward shards the
token dimension before transformer blocks run. TaylorSeer therefore caches each
rank's local token shard, matching the intended multi-GPU memory reduction.

This repository instance currently has no visible GPU, so multi-GPU execution has
not been runtime-validated here. Before formal experiments on the target
multi-GPU machine, run a short smoke test with small `frame_num` and
`sample_steps`, then inspect the TaylorSeer cache summary and CUDA peak memory.

## Single-GPU Example

```bash
/hy-tmp/miniconda3/envs/Wan2.2/bin/python -m third_party.taylorseer_wan22.generate_t2v \
  --task t2v-A14B \
  --ckpt_dir /hy-tmp/models/Wan2.2-T2V-A14B \
  --size 832*480 \
  --frame_num 45 \
  --sample_steps 50 \
  --sample_solver dpm++ \
  --base_seed 42 \
  --offload_model True \
  --convert_model_dtype \
  --save_file /hy-tmp/taylorseer_wan22_smoke.mp4 \
  --taylorseer_fresh_threshold 5 \
  --taylorseer_max_order 1 \
  --taylorseer_first_enhance 1
```

The full official-style method is expected to exceed a single A100 80GB at the
default 832x480/45-frame setting.

## Multi-GPU Example

```bash
torchrun --nproc_per_node=8 -m third_party.taylorseer_wan22.generate_t2v \
  --task t2v-A14B \
  --ckpt_dir /hy-tmp/models/Wan2.2-T2V-A14B \
  --size 832*480 \
  --frame_num 45 \
  --sample_steps 50 \
  --sample_solver dpm++ \
  --base_seed 42 \
  --dit_fsdp \
  --t5_fsdp \
  --ulysses_size 8 \
  --save_file /hy-tmp/taylorseer_wan22_multigpu.mp4 \
  --taylorseer_fresh_threshold 5 \
  --taylorseer_max_order 1 \
  --taylorseer_first_enhance 1
```

## VBench Batch Experiment

Experiment orchestration lives under `experiments/`, not this third-party method
directory:

```text
experiments/taylorseer_vbench_50step_45f_480p/
```

Use that experiment directory for VBench prompt sweeps, artifact archiving,
tmux launch, ffprobe collection, result tables, and failed-run records.
