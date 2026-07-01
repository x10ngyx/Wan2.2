# Session 2026-06-30 MiniDiT Split Compare Runner Launch

## Request

Run a small online comparison of MiniDiT adaptive SeaCache predictors trained with:

- normal sample split
- row split

Test on:

- VBench10
- OpenVid100 training distribution

Use:

- target PSNRs `22` and `28`
- `3` prompts per dataset
- total candidates: `2 models * 2 targets * 2 datasets * 3 prompts = 24`
- reuse existing no-cache baselines; do not spend time regenerating baselines
- write a batch runner and avoid cumulative GPU memory/cache-state growth

## Files Added

- `experiments/adaptive_seacache_mini_dit_split_compare_50step_45f_480p/run_batch.py`
- `experiments/adaptive_seacache_mini_dit_split_compare_50step_45f_480p/run_tmux.sh`

## Runner Design

- Loads the WanT2V pipeline once in one Python process.
- Reuses existing baseline videos/timing:
  - VBench10 dpm++ baselines from `/hy-tmp/wan22_zeus_vbench10_50step_45f_480p_20260624_003030/results/summary.csv`.
  - OpenVid100 train baselines from `/hy-tmp/openvid_100_seacache_trace_data/...`.
- Selects exactly:
  - VBench10: `vbench10_001`, `vbench10_002`, `vbench10_003`
  - OpenVid train: `openvid_002/openvidhd_part1_001`, `openvid_004/openvidhd_part1_003`, `openvid_005/openvidhd_part1_004`
- Compares checkpoints:
  - sample split: `/hy-tmp/wan22_adaptive_threshold_mini_dit_cls_convpatch_3x12x8_d96_l2_bs128_20260629_214906/best_model_checkpoint.pt`
  - row split: `/hy-tmp/wan22_adaptive_threshold_mini_dit_cls_convpatch_rowsplit_packed_d96_l2_bs128_20260629_232659/best_model_checkpoint.pt`
- For each candidate, writes:
  - video
  - log
  - ffprobe JSON
  - PSNR JSON/log
  - adaptive trace JSON/CSV
  - rolling `results/summary.csv`
  - rolling `results/aggregate_by_dataset_model_target.csv`

## Memory Hygiene

The runner handles adaptive SeaCache state per candidate:

- Creates a fresh adaptive SeaCache factory per candidate.
- Assigns that factory to `wan.text2video.SeaCacheTimestepCache`.
- After generation, extracts summary/trace before clearing.
- Calls `clear_last_instance()`.
- Restores the original `SeaCacheTimestepCache`.
- Deletes the factory reference.
- Calls `torch.cuda.empty_cache()`.

This is intended to avoid retaining SeaCache runtime tensors such as previous features, residuals, and live latent snapshots across candidates.

## Validation

- `py_compile` passed for `run_batch.py`.
- CPU validation passed:
  - 6 selected records
  - 2 models
  - 2 target PSNRs
  - exactly 24 expected candidates
  - all baseline videos resolved
  - both checkpoints exist
- GPU was checked before launch:
  - A100 80GB available
  - 0 MiB used
  - no tmux server running before launch

## Launch

Started tmux run:

- tmux session: `wan22_adaptive_mini_dit_split_20260630_025328`
- result root: `/hy-tmp/wan22_adaptive_seacache_mini_dit_split_compare_50step_45f_480p_20260630_025328`
- runner log: `/hy-tmp/wan22_adaptive_seacache_mini_dit_split_compare_50step_45f_480p_20260630_025328/logs/runner.log`
- attach command: `tmux attach -t wan22_adaptive_mini_dit_split_20260630_025328`

## First Candidate Check

First candidate completed successfully:

- dataset: `vbench10`
- sample: `vbench10_001`
- model split: `sample_split`
- target PSNR: `22`
- video: `/hy-tmp/wan22_adaptive_seacache_mini_dit_split_compare_50step_45f_480p_20260630_025328/adaptive_seacache/vbench10/sample_split/target_22/vbench10_001.mp4`
- compute elapsed: `283.984s`
- baseline compute: `538.211s`
- speedup: `1.895x`
- FFmpeg PSNR: `14.928 dB`
- trace rows: `100`
- reuse decisions: `50`
- recompute decisions: `50`
- threshold mean: `0.2725`

This confirms the runner can complete generation, trace extraction, ffprobe, PSNR, and rolling summaries. At session handoff time, the tmux experiment was still running and had moved to `vbench10_001`, sample split, target `28`.
