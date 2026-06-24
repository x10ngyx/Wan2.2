# Adaptive SeaCache Ali Prompt 1-2

Runs timestep-only adaptive SeaCache on Ali prompts 1 and 2 with target PSNR
values `20`, `25`, and `30`.

The runner loads WanT2V once and then sequentially runs all candidates in the
same process. For each candidate it archives:

- generated video;
- ffprobe JSON;
- FFmpeg PSNR against the matching no-cache baseline;
- raw run log;
- command record;
- per-step adaptive trace JSON and CSV containing predicted threshold,
  SeaCache rel-L1, accumulated rel-L1, and reuse/recompute decision.

Cache lifecycle requirement: adaptive SeaCache keeps GPU tensors in
per-candidate runtime state, so the runner must release the latest cache
instance after writing trace/summary and before `torch.cuda.empty_cache()`.
Never retain historical cache instances across candidates.

Launch:

```bash
bash experiments/adaptive_seacache_ali_prompt12_50step_45f_480p/run_tmux.sh
```
