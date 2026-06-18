# Remote Handoff Environment And Git

- User corrected the handoff workflow: upload only the reusable runtime environment to OSS; keep source code in GitHub and use pull/push on the remote machine.
- Abandoned and removed the temporary `/hy-tmp/wan22_current_repo_handoff_build` package plan.
- Reused packed environment from `/hy-tmp/wan22_openvid_first50_handoff_build/Wan2.2/env/Wan2.2-conda-env.tar.gz`.
- Environment archive SHA256: `348f63583d2a3ea742b80341dbb97043c6a497065e593a1329b1aad1a0551f03`.
- Uploaded environment archive to `oss://datasets/Wan2.2-conda-env.tar.gz`.
- Current dirty worktree before commit consisted of `test_sets` Vbench10 prompt-set additions and combined index updates.
- Committed and pushed current code to `x10ngyx/main` at commit `6f68c87`.
