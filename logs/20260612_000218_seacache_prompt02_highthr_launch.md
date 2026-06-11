# 2026-06-12 SeaCache prompt-02 high-threshold launch

- User requested one or two higher SeaCache thresholds for prompt 02.
- Chose thresholds `0.60 0.80` to extend the previous prompt 02 dense sweep beyond `0.50`.
- Preflight:
  - Read `PROGRESS.md`.
  - Confirmed A100 80GB GPU was idle.
  - Confirmed no tmux session was running.
  - `run_batch.py --cpu_validate --prompt_start 1 --prompt_limit 1 --thresholds '0.60 0.80'` passed and found complete reusable prompt 02 baseline artifacts.
- Launched tmux session `seacache_prompt02_highthr_20260612_000218`.
- Result root: `/hy-tmp/wan22_seacache_prompt02_highthr_20260612_000218`.
- Workspace symlink: `experiment_results/wan22_seacache_prompt02_highthr_20260612_000218`.
- Settings: prompt 02 only, seed `42`, `832*480`, 45 frames, 50 DPM++ steps, SeaCache timestep cache only, block cache disabled, CFG cache disabled, reused no-cache baseline from `/hy-tmp/wan22_zeus_threshold_reuse_interp_10prompt_5th_20260608_195427`.
- Launch monitoring confirmed metadata/threshold env files, copied baseline artifacts, and `logs/seacache_th_0p60_prompt_02.log` were created. No failed records were present; GPU memory had risen to about `12GB`, indicating the first candidate had entered generation.
- Existing dirty working tree note: unrelated OpenVid handoff files and `experiments/seacache_openvid100_50step_45f_480p/` were already modified/untracked at session start and were not touched by this launch.

## Completion Check

- tmux exited normally and GPU returned idle.
- No failed records were present.
- Both candidate videos validated with ffprobe as `832x480`, 45 frames, 16 fps, duration `2.812500s`.
- Results:
  - `0.60`: `168.704s`, `3.098x`, PSNR `20.262 dB`, min PSNR `18.27`, reuse/recompute `37/13`.
  - `0.80`: `149.343s`, `3.499x`, PSNR `18.631 dB`, min PSNR `17.42`, reuse/recompute `39/11`.
- Takeaway: `0.60` is the practical upper bound candidate if very high speed is needed; `0.80` is faster but quality drops below the prompt-02 `0.50` result by about `5.09 dB`.
