# Session Log 2026-06-25 VS Code Remote Stability

- Investigated repeated VS Code Remote disconnects and failed Git askpass authentication.
- Confirmed system resources were healthy: root disk, `/hy-tmp`, memory, and inode usage were not exhausted.
- Found VS Code Remote `remoteagent.log` entries showing repeated Extension Host `SIGKILL` after startup, with Python/Pylance, GitHub Copilot Chat, Anyscale, and Python environment extensions activating.
- Added `.vscode/settings.json` to exclude large experiment/model/cache symlinks and disable expensive Python/Pylance indexing/test discovery for this workspace.
- Updated `.gitignore` to allow tracking `.vscode/settings.json` while keeping other `.vscode` files ignored.
- Reconnected and observed the Extension Host remaining alive for more than six minutes; the prior rapid `SIGKILL` loop did not recur during the check.
- No inference, PSNR, or dataset jobs were run.
