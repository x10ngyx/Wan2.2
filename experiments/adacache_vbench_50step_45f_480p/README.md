# AdaCache VBench 50-step 45f 480p

Single-process Wan2.2 T2V runner for the isolated AdaCache adapter in
`third_party/AdaCache`.

The runner loads the WanT2V pipeline once, then sequentially runs each selected
VBench prompt with:

- no-cache baseline;
- AdaCache candidate using the official default-style codebook.

It archives videos, ffprobe JSON, PSNR JSON/logs, raw generation logs, command
records, manifests, failed records, and summary tables under `/hy-tmp`.

## Defaults

- task: `t2v-A14B`
- checkpoint: `/hy-tmp/models/Wan2.2-T2V-A14B`
- prompts: `test_sets/vbench_every20/prompts.jsonl`
- seed: `42`
- size: `832*480`
- frame count: `45`
- sample steps: `50`
- solver: `dpm++`
- offload: enabled
- dtype conversion: enabled
- AdaCache residual: `t-attn`
- AdaCache cache location: `13`
- AdaCache codebook: `0.03:12,0.05:10,0.07:8,0.09:6,0.11:4,1.0:3`
- MoReg: disabled by default

## CPU Validation

```bash
/hy-tmp/miniconda3/envs/Wan2.2/bin/python \
  experiments/adacache_vbench_50step_45f_480p/run_batch.py \
  --cpu_validate
```

## Launch

```bash
bash experiments/adacache_vbench_50step_45f_480p/run_tmux.sh
```

For a short smoke run:

```bash
/hy-tmp/miniconda3/envs/Wan2.2/bin/python \
  experiments/adacache_vbench_50step_45f_480p/run_batch.py \
  --prompt_limit 1 \
  --convert_model_dtype
```

Enable MoReg:

```bash
/hy-tmp/miniconda3/envs/Wan2.2/bin/python \
  experiments/adacache_vbench_50step_45f_480p/run_batch.py \
  --adacache_moreg \
  --convert_model_dtype
```
