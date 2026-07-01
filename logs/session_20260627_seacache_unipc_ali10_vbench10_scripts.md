# 2026-06-27 SeaCache UniPC Ali-10 / VBench10 Scripts

Task: switch sampler to `unipc`, keep other SeaCache experiment settings unchanged, and prepare separate Ali-10 and VBench10 experiment scripts with queued execution.

Added:

- `experiments/seacache_unipc_ali10_50step_45f_480p/README.md`
- `experiments/seacache_unipc_ali10_50step_45f_480p/run_tmux.sh`
- `experiments/seacache_unipc_vbench10_50step_45f_480p/README.md`
- `experiments/seacache_unipc_vbench10_50step_45f_480p/run_tmux.sh`
- `experiments/seacache_unipc_queue_ali10_vbench10_50step_45f_480p.sh`

Implementation notes:

- The wrappers reuse the existing single-process runner:
  `experiments/seacache_vbench10_50step_45f_480p/run_batch.py`.
- Both pass `--sample_solver unipc`.
- Ali-10 wrapper uses `test_sets/ali_10/prompts.jsonl`.
- VBench10 wrapper uses `test_sets/Vbench10/prompts.jsonl`.
- Queue script starts Ali-10 first, waits for its tmux session to finish, then starts VBench10.
- Defaults preserve the existing SeaCache settings: seed 42, `832*480`, 45 frames, 50 steps, timestep cache `seacache`, no block cache, no CFG cache, and thresholds `0.10 0.20 0.30 0.50`.

Validation:

- `bash -n experiments/seacache_unipc_ali10_50step_45f_480p/run_tmux.sh experiments/seacache_unipc_vbench10_50step_45f_480p/run_tmux.sh experiments/seacache_unipc_queue_ali10_vbench10_50step_45f_480p.sh`
- `/hy-tmp/miniconda3/envs/Wan2.2/bin/python experiments/seacache_vbench10_50step_45f_480p/run_batch.py --cpu_validate --prompt_path test_sets/ali_10/prompts.jsonl --sample_solver unipc`
- `/hy-tmp/miniconda3/envs/Wan2.2/bin/python experiments/seacache_vbench10_50step_45f_480p/run_batch.py --cpu_validate --prompt_path test_sets/Vbench10/prompts.jsonl --sample_solver unipc`
- After narrowing thresholds:
  - `/hy-tmp/miniconda3/envs/Wan2.2/bin/python experiments/seacache_vbench10_50step_45f_480p/run_batch.py --cpu_validate --prompt_path test_sets/ali_10/prompts.jsonl --sample_solver unipc --thresholds '0.10 0.20 0.30 0.50'`
  - `/hy-tmp/miniconda3/envs/Wan2.2/bin/python experiments/seacache_vbench10_50step_45f_480p/run_batch.py --cpu_validate --prompt_path test_sets/Vbench10/prompts.jsonl --sample_solver unipc --thresholds '0.10 0.20 0.30 0.50'`

Initial CPU validations used the old 10-threshold set. After the user narrowed the sweep to `0.10 0.20 0.30 0.50`, the wrappers were updated to run 10 expected baselines and 40 expected SeaCache candidates per prompt set.

Run status:

- Initially not launched because GPU mode was unavailable: `nvidia-smi` returned `No devices were found`.
- Later launched after GPU became available.
- Queue session: `wan22_seacache_unipc_queue_20260627_023222`.
- Ali-10 session: `wan22_seacache_unipc_queue_20260627_023222_ali10`.
- Ali-10 result root: `/hy-tmp/wan22_seacache_unipc_ali10_50step_45f_480p_20260627_023222`.
- VBench10 result root, queued second: `/hy-tmp/wan22_seacache_unipc_vbench10_50step_45f_480p_20260627_023222`.
- Launch check showed `NVIDIA A100 80GB PCIe` idle at start; Ali-10 runner started and was loading WanT2V checkpoint shards.

Baseline reuse follow-up:

- User asked whether old ZEUS UniPC baseline artifacts could be reused.
- The queue was stopped before continuing baseline generation.
- Reports confirm the old ZEUS UniPC baseline parameters match this experiment, but the expected local result roots are currently missing:
  - `/hy-tmp/wan22_zeus_unipc_ali10_50step_45f_480p_20260624_195011`
  - `/hy-tmp/wan22_zeus_unipc_vbench10_50step_45f_480p_20260624_192306`
- The corresponding `experiment_results/wan22_zeus_unipc_*` symlinks are dangling, and the downloaded VBench archives do not contain these UniPC baseline videos/logs/ffprobe artifacts.
- Existing VBench baseline archives are DPM++ baselines, not UniPC, so they are not valid PSNR references for this UniPC run.
- The initial launch had completed `ali_001` baseline artifacts. The queue was restarted with the same result roots and `--resume_existing`, so `ali_001` can be reused from the partial Ali-10 result root while the remaining missing baselines/candidates run.
- Restarted queue session: `wan22_seacache_unipc_queue_20260627_023222`; Ali-10 child session restarted at 2026-06-27 02:54.

Corrective update:

- User pointed out the initial wrappers did not satisfy the separate batch-runner experiment-script requirement because they called the existing VBench10 SeaCache runner directly.
- Stopped the queue again and added dedicated formal runners/summarizers:
  - `experiments/seacache_unipc_ali10_50step_45f_480p/run_batch.py`
  - `experiments/seacache_unipc_ali10_50step_45f_480p/summarize_results.py`
  - `experiments/seacache_unipc_vbench10_50step_45f_480p/run_batch.py`
  - `experiments/seacache_unipc_vbench10_50step_45f_480p/summarize_results.py`
- Fixed wrappers to call their local `run_batch.py` files.
- Fixed runner/report naming:
  - manifest files are now `selected_records.jsonl` and `selected_records.csv`;
  - summaries write `sample_solver=unipc`;
  - default roots are `wan22_seacache_unipc_ali10_...` and `wan22_seacache_unipc_vbench10_...`.
- Validation after correction:
  - `python -m py_compile` passed for both new runners and summarizers.
  - CPU validation passed for both new runners with 10 prompts, 4 thresholds, and 40 expected candidates.
  - `bash -n` and `git diff --check` passed.
- Queue restarted at 2026-06-27 03:14 with the corrected dedicated Ali-10 runner. Process command confirmed:
  `experiments/seacache_unipc_ali10_50step_45f_480p/run_batch.py`.

Monitoring:

- Added `experiments/seacache_unipc_monitor_ali10_vbench10_50step_45f_480p.sh`.
- Launched tmux monitor session `wan22_seacache_unipc_monitor_20260627_023222`.
- Monitor interval: 600 seconds.
- Monitor log: `/hy-tmp/wan22_seacache_unipc_ali10_50step_45f_480p_20260627_023222/logs/queue_monitor.log`.
- First monitor record showed:
  - queue and Ali-10 tmux sessions active;
  - GPU at about `47351 MiB`, `100%`;
  - Ali-10 had 1 baseline MP4 and 1 completed SeaCache candidate from resumed partial output;
  - 0 failed files;
  - VBench10 had not started yet.

VBench10 progress check at 2026-06-27 14:41 CST:

- tmux sessions active:
  - `wan22_seacache_unipc_monitor_20260627_023222`
  - `wan22_seacache_unipc_queue_20260627_023222`
  - `wan22_seacache_unipc_queue_20260627_023222_vbench10`
- GPU active: `NVIDIA A100 80GB PCIe`, about `47345 MiB`, `100%`.
- VBench10 artifacts:
  - baseline MP4: `10/10`
  - SeaCache candidate MP4: `38/40`
  - candidate time files: `38/40`
  - PSNR JSON: `38/40`
  - ffprobe JSON: `48/50`
  - failed files: `0`
- Missing candidates:
  - `th_0p30/vbench10_010.mp4`
  - `th_0p50/vbench10_010.mp4`
- Runner log showed `vbench10_010` in progress. Summary and aggregate tables had not yet been generated because the batch was still running.
