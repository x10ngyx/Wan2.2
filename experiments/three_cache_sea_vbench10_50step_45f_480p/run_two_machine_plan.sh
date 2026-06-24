#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
EXP_PREFIX="${EXP_PREFIX:-wan22_three_cache_sea_vbench10_50step_45f_480p_${STAMP}}"
PROMPT_PATH="${PROMPT_PATH:-${ROOT_DIR}/test_sets/Vbench10/prompts.jsonl}"
TIMESTEP_THRESHOLDS="${TIMESTEP_THRESHOLDS:-0.05 0.10 0.20 0.50}"
BLOCK_THRESHOLDS="${BLOCK_THRESHOLDS:-0.05 0.10 0.20 0.50}"
CFG_THRESHOLDS="${CFG_THRESHOLDS:-0.05 0.10 0.20 0.50}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-0}"

cat <<EOF
# Run this on machine A for VBench10 prompts 1-5:
STAMP='${STAMP}' \\
EXP_ROOT='/hy-tmp/${EXP_PREFIX}_machineA_p000_004' \\
SESSION='wan22_3sea_vbench10_${STAMP}_p000_004' \\
CUDA_VISIBLE_DEVICES_VALUE='${CUDA_VISIBLE_DEVICES_VALUE}' \\
PROMPT_PATH='${PROMPT_PATH}' \\
PROMPT_START=0 \\
PROMPT_LIMIT=5 \\
TIMESTEP_THRESHOLDS='${TIMESTEP_THRESHOLDS}' \\
BLOCK_THRESHOLDS='${BLOCK_THRESHOLDS}' \\
CFG_THRESHOLDS='${CFG_THRESHOLDS}' \\
bash '${ROOT_DIR}/experiments/three_cache_sea_vbench10_50step_45f_480p/run_tmux.sh'

# Run this on machine B for VBench10 prompts 6-10:
STAMP='${STAMP}' \\
EXP_ROOT='/hy-tmp/${EXP_PREFIX}_machineB_p005_009' \\
SESSION='wan22_3sea_vbench10_${STAMP}_p005_009' \\
CUDA_VISIBLE_DEVICES_VALUE='${CUDA_VISIBLE_DEVICES_VALUE}' \\
PROMPT_PATH='${PROMPT_PATH}' \\
PROMPT_START=5 \\
PROMPT_LIMIT=5 \\
TIMESTEP_THRESHOLDS='${TIMESTEP_THRESHOLDS}' \\
BLOCK_THRESHOLDS='${BLOCK_THRESHOLDS}' \\
CFG_THRESHOLDS='${CFG_THRESHOLDS}' \\
bash '${ROOT_DIR}/experiments/three_cache_sea_vbench10_50step_45f_480p/run_tmux.sh'

# After both shards are copied under one parent directory, merge with:
python '${ROOT_DIR}/experiments/three_cache_sea_vbench10_50step_45f_480p/merge_shards.py' \\
  --parent-root '/hy-tmp/${EXP_PREFIX}_merged_parent'
EOF
