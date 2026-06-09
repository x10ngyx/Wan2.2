# 2026-06-09 block-group window params and restart

- User requested block-group cache script parameters enable first/last segment protection and max consecutive reuse.
- Updated `experiments/block_cache_only_50step_45f_480p/run_experiments.sh` defaults:
  - `BLOCK_GROUP_START=0.1`
  - `BLOCK_GROUP_END=0.9`
  - `BLOCK_GROUP_MAX_REUSE=3` remains enabled.
- Updated the experiment README to document the protected block-group window and max reuse setting.
- Validation: `bash -n experiments/block_cache_only_50step_45f_480p/run_experiments.sh` passed.
- Stopped the previous tmux retry and archived partial `bwcache_th_0p15` logs under the experiment root's `failed_history/`.
- Restarted the same result root in tmux session `block_cache_only_p01_retry5_windowed` with `RESUME_EXISTING=True`.
- New script-level env confirms `block_group_start=0.1`, `block_group_end=0.9`, and `block_group_max_reuse=3`.
- Completed result check before restart:
  - baseline prompt 01: video, ffprobe JSON, log, and compute time present.
  - `bwcache_th_0p05` prompt 01: video, ffprobe JSON, PSNR JSON/log, raw log, command, and compute time present.
  - baseline compute time `522.603s`; `bwcache_th_0p05` compute time `507.238s`, speedup `1.030x`.
  - `bwcache_th_0p05` FFmpeg mean PSNR `28.895 dB`.
  - Both baseline and `bwcache_th_0p05` ffprobe metadata: `832x480`, `45` frames, `16 fps`, duration `2.812500s`.
