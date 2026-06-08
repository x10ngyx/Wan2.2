# 2026-06-08 Repo Familiarization

- Read `PROGRESS.md` and followed AGENTS workflow expectations.
- Reviewed repository layout, README, and the current T2V path: `generate.py`, `wan/text2video.py`, and `wan/timestep_cache.py`.
- Confirmed current HEAD includes native T2V ZEUS timestep cache and `zeus-threshold` timestep cache support.
- Confirmed `--block_cache none` and `--cfg_cache none` are still placeholder interfaces; no block cache or CFG cache behavior is implemented yet.
- Reviewed experiment scripts under `experiments/zeus_timestep_cache_50step_45f_480p/` and `experiments/zeus_threshold_50step_45f_480p/`; both archive outputs under `/hy-tmp` with commands, logs, ffprobe, PSNR, summaries, and failure records.
- Noted environment caveat from progress history: full generation/PSNR validation requires GPU mode; previous notes said no GPU was visible.
