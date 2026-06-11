# 2026-06-10 repo and ablation results read

- Read `PROGRESS.md`, recent logs, current repository file layout, git status, and latest `experiment_results` symlinks.
- Current working tree is dirty with cache implementation changes, ablation experiment files, recent logs, and result symlinks; this session did not modify experiment code.
- Checked tmux: no tmux server is running, so the latest ablation session has exited.
- Latest result root: `/hy-tmp/wan22_cache_ablation_prompt01_50step_45f_480p_20260609_184625`.
- The ablation run completed all 7 prompt 01 candidates with no active failed records: timestep only, block only, CFG only, timestep+block, timestep+CFG, block+CFG, and all three.
- Result archive contains videos, configs, logs, compute-only `.time` files, ffprobe JSON, FFmpeg PSNR outputs, `results/summary.csv`, and `results/aggregate.json`.
- Baseline compute time is `522.603s`. Candidate summary: timestep only `1.600x` / `18.606 dB`; block only `1.362x` / `19.396 dB`; CFG only `1.148x` / `21.571 dB`; timestep+block `1.748x` / `18.159 dB`; timestep+CFG `1.332x` / `20.910 dB`; block+CFG `1.352x` / `19.446 dB`; all three `1.370x` / `19.603 dB`.
- All candidate videos ffprobe-validate as `832x480`, `45` frames, `16 fps`, duration `2.812500s`.
- Key reading: CFG alone is not the main quality-drop source in this parameter set; timestep threshold `0.02` is the strongest driver of low PSNR, while CFG can partially recover quality when combined with timestep at a speed cost.
