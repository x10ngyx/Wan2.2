# Session 2026-07-04 SeaCache Prompt-02 Parameter Check

## Request

Check why the DPM++ prompt-02 SeaCache pilot looks significantly better than the complete Ali-10 prompt-02 result.

## Checked Artifacts

- DPM++ dense pilot: `/hy-tmp/wan22_seacache_prompt02_dense_20260611_204826`
- DPM++ high-threshold pilot: `/hy-tmp/wan22_seacache_prompt02_highthr_20260612_000218`
- Complete Ali-10 UniPC run: `/hy-tmp/wan22_seacache_unipc_ali10_50step_45f_480p_20260627_023222`
- DPM++ reused baseline root: `/hy-tmp/wan22_zeus_threshold_reuse_interp_10prompt_5th_20260608_195427`

## Parameter Findings

Prompt text, seed, size, frame count, sample steps, sample shift, guide scale,
SeaCache parameters, block cache disabled, and CFG cache disabled match between
the DPM++ prompt-02 pilot and the UniPC Ali-10 `ali_002` run.

The material difference is:

- DPM++ pilot: `sample_solver=dpm++`
- Complete Ali-10 run: `sample_solver=unipc`

The DPM++ pilot also reuses the matching DPM++ baseline from the older ZEUS
threshold experiment rather than generating a baseline inside the SeaCache pilot
root. This is valid for that pilot because prompt/seed/shape/solver settings
match the reused baseline.

## Metrics Spot Check

At threshold `0.20`, both runs reuse `20` timesteps and recompute `30`
timesteps, with nearly identical speedup:

- DPM++ prompt-02: PSNR `30.097 dB`, speedup `1.562x`
- UniPC `ali_002`: PSNR `29.805 dB`, speedup `1.577x`

At more aggressive thresholds the solver difference matters more:

- threshold `0.30`: DPM++ `29.582 dB`, UniPC `24.790 dB`
- threshold `0.50`: DPM++ `23.725 dB`, UniPC `18.252 dB`

## Conclusion

The DPM++ pilot is not better because of lower reuse or different SeaCache
configuration. It is better on prompt-02 mainly because DPM++ is more robust to
the same SeaCache reuse pattern than UniPC for this prompt.

## Changes

- Updated `PROGRESS.md` with a concise parameter-check note.
- Added this session log.

## Validation

- Read command records, runner config JSON/logs, prompt files, and summary CSVs.
- No inference or GPU validation was run.
