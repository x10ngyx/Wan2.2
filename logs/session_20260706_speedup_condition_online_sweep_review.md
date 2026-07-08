# Session 2026-07-06 Speedup-Conditioned Online Sweep Review

Context: checked the transformer/MiniDiT predictor online inference experiment after adding `target_speedup`.

Reviewed:
- New online sweep root: `/hy-tmp/wan22_adaptive_seacache_mini_dit_rowsplit_speedup_sweep_50step_45f_480p_20260706_194715`
- Old comparison root: `/hy-tmp/wan22_adaptive_seacache_mini_dit_split_compare_50step_45f_480p_20260630_025328`
- Runner: `experiments/adaptive_seacache_mini_dit_split_compare_50step_45f_480p/run_batch.py`
- Online gate path: `adaptive_seacache_wan22/cache.py`
- Speedup-conditioned row-split checkpoint: `/hy-tmp/wan22_adaptive_threshold_mini_dit_cls_convpatch_rowsplit_packed_d96_l2_bs128_speedup_20260706_171523/best_model_checkpoint.pt`

Findings:
- No runner crash or missing `target_speedup` plumbing found. The checkpoint has `cond_embed.0.weight` shape `(96, 3)`, and the online loader logs the requested `target_speedup`.
- `--cpu_validate` passed for the new sweep and reported `36` expected candidates.
- New and old online runs use the same task, size, frame count, sample steps, solver, shift, guide scale, seed, prompt records, and baseline videos.
- The new run is not a full same-matrix comparison with the old run: it runs only `row_split` and sweeps target speedup by target PSNR, while the old run compared `sample_split` and `row_split` with one implicit speedup condition per PSNR target.
- First completed point wrote the full archive chain. Result: `vbench10_001`, target PSNR `22`, target speedup `2.2`, speedup `2.262x`, PSNR `15.176 dB`, threshold mean `0.362`, predictor mean time about `0.0051s`.
- Early effect signal is poor versus target PSNR and versus the old row-split target-22 point on the same sample, but the full `36`-candidate run is still in progress.

Validation:
- Ran static config comparison between new and old experiment configs.
- Ran runner `--cpu_validate`.
- Checked GPU/tmux status without interrupting the active run.

Unfinished:
- Wait for tmux session `wan22_adaptive_seacache_mini_dit_rowsplit_speedup_sweep_50step_45f_480p_20260706_194715` to finish, then aggregate all 36 candidates and compare against the old row-split MiniDiT online result and fixed SeaCache controls.
