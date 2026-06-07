# ZEUS Timestep Cache T2V Experiment

Configuration:

- Model: Wan2.2 T2V-A14B
- Checkpoint: `/hy-tmp/models/Wan2.2-T2V-A14B`
- Prompts: `/hy-tmp/work/Wan2.2/prompt.txt` (10 prompts)
- Steps: 50
- Frames: 45
- Resolution: 480p landscape (`832*480`)
- Solver: `dpm++` by default, override with `SAMPLE_SOLVER=unipc`
- Methods: baseline (`--timestep_cache none`) and ZEUS (`--timestep_cache zeus`)

Run:

```bash
bash experiments/zeus_timestep_cache_50step_45f_480p/run_experiments.sh
```

The script writes videos, commands, raw logs, `ffprobe` JSON, PSNR JSON/logs, and CSV summaries under `/hy-tmp`.

## ZEUS Implementation Match

The timestep-cache rule uses the same public ZEUS Wan/FlowMatch parameters and scheduling controls:

- `acc_range`
- `denominator`
- `modular`
- `caching_mode`
- `max_interval`
- `lagrange_term`
- `lagrange_int`
- `lagrange_step`

The native Wan2.2 T2V path differs from the ZEUS demo in two structural ways:

- Wan2.2 A14B switches between `high_noise_model` and `low_noise_model`.
- This repository calls cond and uncond branches separately.

Therefore the implementation keeps independent ZEUS states for `(model_stage, branch)`:

- `("high", "cond")`
- `("high", "uncond")`
- `("low", "cond")`
- `("low", "uncond")`

Within each state, the ZEUS skip schedule and reuse/interpolation controls follow the official ZEUS Wan/FlowMatch path. State separation is the required adaptation that prevents cross-model or cross-branch output reuse.
