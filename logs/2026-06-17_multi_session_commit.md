# 2026-06-17 Multi-session Commit

- User requested a git commit for accumulated progress from multiple sessions.
- Read `PROGRESS.md` at session start per project instructions.
- Inspected `git status --short`, `git diff --stat`, `PROGRESS.md` diff, deleted `taylorseer_wan22/` diff, untracked files, and size checks for new directories.
- Confirmed the visible changes match the recorded work:
  - TaylorSeer standalone integration moved from top-level `taylorseer_wan22/` to `third_party/taylorseer_wan22/`.
  - Added adaptive SeaCache prototype under `adaptive_seacache_wan22/`.
  - Added TaylorSeer, adaptive SeaCache, and AdaCache experiment runners under `experiments/`.
  - Added copied AdaCache source and Wan2.2 adapter under `third_party/AdaCache/`.
  - Added 2026-06-16 implementation, experiment, and training logs.
  - Added `todo.md` with next work items.
- No new tests or GPU jobs were run in this commit-only session.
- Ignored `__pycache__/` files were left unstaged.
