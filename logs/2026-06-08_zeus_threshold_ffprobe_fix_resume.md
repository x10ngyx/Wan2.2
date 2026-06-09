# 2026-06-08 ZEUS-threshold ffprobe fix and resume

- Status check at 17:59 CST found tmux had exited again after completing threshold 0.001 generation.
- Root cause: ffprobe was not found in the tmux PATH. The script was running plain ffprobe under set -e, which exited immediately after leaving 0-byte ffprobe JSON files and no failed record.
- Updated experiments/zeus_threshold_50step_45f_480p/run_experiments.sh to use FFPROBE_BIN, defaulting to /hy-tmp/miniconda3/envs/Wan2.2/bin/ffprobe. Resume checks now require nonempty metadata/PSNR files. bash -n passed.
- Regenerated nonempty ffprobe JSON for baseline prompt 01 and threshold 0.001, and computed PSNR for threshold 0.001. Since threshold 0.001 had no cache reuse, PSNR excluded all 45 perfect frames and reports Infinity.
- Restarted tmux session zeus_threshold_p01_resume2_1800. It skipped existing baseline and 0.001 artifacts and started threshold 0.005.
