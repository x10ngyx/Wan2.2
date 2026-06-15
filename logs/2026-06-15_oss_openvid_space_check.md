# 2026-06-15 OSS OpenVid Space Check

- Read `PROGRESS.md` at session start per project workflow.
- Checked local disk:
  - `/hy-tmp`: `400G` total, `265G` used, `136G` available.
  - `/`: `30G` total, `7.7G` used, `23G` available.
- Initial `oss ls -s -d oss://datasets/` failed with `401 Authentication Failed`.
- Logged in to OSS with the user-provided HyCloud account and listed `oss://datasets/`.
- Relevant OSS objects:
  - `prompt001-033.tar.gz`: `40.59GB`
  - `prompt034-100.tar.gz`: `82.61GB`
  - Combined compressed size: about `123.20GB`.
- Assessment: `/hy-tmp` can barely hold both compressed prompt archives, but doing so would leave about `12-13GB` free and would not leave room to extract them. Recommended options are to free space first, download one shard at a time, or use selective/streaming extraction instead of storing compressed and extracted copies simultaneously.
- User then requested downloading both archives anyway, using tmux.
- Added `scripts/download_openvid_prompt_archives.sh` and launched tmux session `download_openvid_prompts`.
- Download log: `logs/2026-06-15_openvid_prompt_archive_download.log`.
- Launch check showed one active `oss cp` for `prompt001-033.tar.gz`; no extraction was started.
