# AdaCache VBench10 Slow/Fast

Runs VBench10 on two GPUs with three outputs per prompt:

- no-cache baseline
- AdaCache slow
- AdaCache fast

The baseline is used for both PSNR and speedup. Slow/fast each support separate
high-noise and low-noise Wan2.2 codebooks.

Defaults:

- prompts: `test_sets/Vbench10/prompts.jsonl`
- checkpoint: `/hy-tmp/models/Wan2.2-T2V-A14B`
- python: `/hy-tmp/env/Wan2.2-fa2torch28/bin/python`
- ffprobe: `/hy-tmp/env/Wan2.2-fa2torch28/bin/ffprobe`
- ffmpeg: `/hy-tmp/env/Wan2.2-fa2torch28/bin/ffmpeg`
- slow high/low: `wan22_50_slow` / `wan22_50_slow`
- fast high/low: `wan22_50_fast` / `wan22_50_fast`

CPU validation:

```bash
/hy-tmp/env/Wan2.2-fa2torch28/bin/python \
  experiments/adacache_vbench10_slow_fast_50step_45f_480p/run_batch.py \
  --cpu_validate
```

Launch:

```bash
bash experiments/adacache_vbench10_slow_fast_50step_45f_480p/run_tmux.sh
```

Override stage codebooks:

```bash
SLOW_HIGH_CODEBOOK_PRESET=wan22_50_slow \
SLOW_LOW_CODEBOOK_PRESET=wan22_50_fast \
FAST_HIGH_CODEBOOK_PRESET=wan22_50_fast \
FAST_LOW_CODEBOOK_PRESET=official_100 \
bash experiments/adacache_vbench10_slow_fast_50step_45f_480p/run_tmux.sh
```
