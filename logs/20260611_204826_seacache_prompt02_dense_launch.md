# 2026-06-11 SeaCache prompt-02 dense launch

- Checked the completed prompt-01 SeaCache pilot before launching another run. It finished with no failures and wrote summary/aggregate results under `/hy-tmp/wan22_seacache_50step_45f_480p_20260611_191733`.
- Prompt-01 takeaway: threshold `0.10` gave high PSNR with modest speedup, `0.20` looked like the most balanced point, and `0.30`/`0.50` traded substantial quality for speed.
- User requested more thresholds on the second prompt.
- Selected denser thresholds around the useful transition region plus aggressive endpoints: `0.08 0.10 0.12 0.15 0.18 0.20 0.25 0.30 0.40 0.50`.
- Preflight checks:
  - GPU idle on A100 80GB.
  - `/hy-tmp` had about `144G` free.
  - No existing tmux session.
  - `run_batch.py --cpu_validate --prompt_start 1 --prompt_limit 1` confirmed complete reusable prompt 02 baseline artifacts.
- Launched tmux session `seacache_prompt02_dense_20260611_204826`.
- Result root: `/hy-tmp/wan22_seacache_prompt02_dense_20260611_204826`.
- Workspace symlink: `experiment_results/wan22_seacache_prompt02_dense_20260611_204826`.
- Settings: prompt 02 only, seed `42`, `832*480`, 45 frames, 50 dpm++ steps, SeaCache timestep cache only, block cache disabled, CFG cache disabled, reused no-cache baseline artifacts from `/hy-tmp/wan22_zeus_threshold_reuse_interp_10prompt_5th_20260608_195427`.
- Launch monitoring confirmed metadata/threshold env files were created, no failed records existed, the shared WanT2V pipeline initialized, checkpoint shards loaded, and the first candidate `SeaCache 0.08 prompt 02` started sampling.

## Completion Check

- The tmux session exited normally and GPU returned idle.
- No failed records were found under the experiment root.
- All 10 SeaCache candidate videos were generated and validated with ffprobe as `832x480`, 45 frames, 16 fps, duration `2.812500s`.
- Summary files are present:
  - `/hy-tmp/wan22_seacache_prompt02_dense_20260611_204826/results/summary.csv`
  - `/hy-tmp/wan22_seacache_prompt02_dense_20260611_204826/results/aggregate_by_threshold.csv`
  - `/hy-tmp/wan22_seacache_prompt02_dense_20260611_204826/results/aggregate_by_threshold.json`
- Baseline compute time: `522.608s`.
- Aggregate results:
  - `0.08`: `528.725s`, `0.988x`, PSNR `Infinity`, reuse/recompute `0/50`.
  - `0.10`: `479.491s`, `1.090x`, PSNR `45.532 dB`, reuse/recompute `5/45`.
  - `0.12`: `421.440s`, `1.240x`, PSNR `42.475 dB`, reuse/recompute `11/39`.
  - `0.15`: `372.696s`, `1.402x`, PSNR `35.441 dB`, reuse/recompute `16/34`.
  - `0.18`: `343.774s`, `1.520x`, PSNR `29.848 dB`, reuse/recompute `19/31`.
  - `0.20`: `334.663s`, `1.562x`, PSNR `30.097 dB`, reuse/recompute `20/30`.
  - `0.25`: `285.500s`, `1.831x`, PSNR `29.055 dB`, reuse/recompute `25/25`.
  - `0.30`: `265.988s`, `1.965x`, PSNR `29.582 dB`, reuse/recompute `27/23`.
  - `0.40`: `217.301s`, `2.405x`, PSNR `27.044 dB`, reuse/recompute `32/18`.
  - `0.50`: `197.847s`, `2.641x`, PSNR `23.725 dB`, reuse/recompute `34/16`.
