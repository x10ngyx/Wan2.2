# 2026-07-06 Row-Split Target-Speedup Online Test

## Scope

- Prepared and launched the online adaptive SeaCache test for the speedup-conditioned MiniDiT/Transformer predictor.
- The run tests only the `row_split` checkpoint.
- No git commit was made.

## Runner Update

- Updated `experiments/adaptive_seacache_mini_dit_split_compare_50step_45f_480p/run_batch.py` to support:
  - `--model_splits`, so the existing split-compare runner can run only `row_split`.
  - `--target_speedups_by_psnr`, so each target PSNR can have multiple speedup conditioning values.
  - speedup labels in method IDs and artifact subdirectories to avoid overwriting results across speedup settings.
  - dynamic expected candidate count for the prior 24-candidate split compare and this 36-candidate row-split speedup sweep.

## Test Design

- Experiment root: `/hy-tmp/wan22_adaptive_seacache_mini_dit_rowsplit_speedup_sweep_50step_45f_480p_20260706_194715`
- Symlink: `experiment_results/wan22_adaptive_seacache_mini_dit_rowsplit_speedup_sweep_50step_45f_480p_20260706_194715`
- tmux session: `wan22_adaptive_seacache_mini_dit_rowsplit_speedup_sweep_50step_45f_480p_20260706_194715`
- tmux log: `/hy-tmp/wan22_adaptive_seacache_mini_dit_rowsplit_speedup_sweep_50step_45f_480p_20260706_194715.tmux.log`
- Checkpoint: `/hy-tmp/wan22_adaptive_threshold_mini_dit_cls_convpatch_rowsplit_packed_d96_l2_bs128_speedup_20260706_171523/best_model_checkpoint.pt`
- Prompt setting: same as the previous MiniDiT split-compare online test, with VBench10 first 3 prompts and OpenVid train first 3 prompts.
- Targets:
  - `target_psnr=22`: `target_speedup=2.2,2.5,2.8`
  - `target_psnr=28`: `target_speedup=1.4,1.7,2.0`
- Candidate count: `6 prompts * 1 split * 2 PSNR targets * 3 speedup settings = 36`.

## Speedup Setting Rationale

- Middle values are the expected speedups estimated from fixed SeaCache PSNR-speed curves plus the previous row-split online test.
- Lower and higher values probe whether the speedup-conditioned predictor moves thresholds and online speed/quality in the expected direction.

## Verification

- `py_compile` passed for the modified runner.
- CPU validation passed with expected candidate count `36`.
- GPU mode was available: `NVIDIA A100 80GB PCIe`.
- The run loaded the row-split speedup-conditioned checkpoint successfully and started the first candidate:
  - `vbench10_vbench10_001_row_split_target_22_speedup_2p2`
  - `target_psnr=22`
  - `target_speedup=2.2`
- During first-candidate sampling, GPU utilization reached `100%` and memory use was about `47GB`.

## Follow-Up

- Wait for the tmux run to complete.
- Then inspect:
  - `/hy-tmp/wan22_adaptive_seacache_mini_dit_rowsplit_speedup_sweep_50step_45f_480p_20260706_194715/results/summary.csv`
  - `/hy-tmp/wan22_adaptive_seacache_mini_dit_rowsplit_speedup_sweep_50step_45f_480p_20260706_194715/results/aggregate_by_dataset_model_target.csv`
- Compare monotonicity across target speedups for each PSNR target: predicted threshold, reuse count, achieved speedup, and PSNR.
