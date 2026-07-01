#!/usr/bin/env bash
set -euo pipefail

cd /hy-tmp/work/Wan2.2

STAMP="$(date +%Y%m%d_%H%M%S)"
EXP_ROOT="/hy-tmp/wan22_adaptive_seacache_mini_dit_split_compare_50step_45f_480p_${STAMP}"
SESSION="wan22_adaptive_mini_dit_split_${STAMP}"

mkdir -p "${EXP_ROOT}/logs"

export HF_HOME=/hy-tmp/hf-cache
export TRANSFORMERS_CACHE=/hy-tmp/hf-cache
export HF_HUB_CACHE=/hy-tmp/hf-cache/hub
export PYTHONUNBUFFERED=1

tmux new-session -d -s "${SESSION}" \
  "/hy-tmp/miniconda3/envs/Wan2.2/bin/python experiments/adaptive_seacache_mini_dit_split_compare_50step_45f_480p/run_batch.py \
    --exp_root '${EXP_ROOT}' \
    --python_bin /hy-tmp/miniconda3/envs/Wan2.2/bin/python \
    --ffprobe_bin /hy-tmp/miniconda3/envs/Wan2.2/bin/ffprobe \
    --target_psnrs '22 28' \
    --prompt_count 3 \
    --resume_existing 2>&1 | tee '${EXP_ROOT}/logs/runner.log'"

echo "tmux_session=${SESSION}"
echo "exp_root=${EXP_ROOT}"
echo "attach=tmux attach -t ${SESSION}"
echo "log=${EXP_ROOT}/logs/runner.log"
