#!/usr/bin/env bash
set -euo pipefail

SESSION=${SESSION:-wan22_adacache_vbench}
ROOT_DIR=${ROOT_DIR:-/hy-tmp/work/Wan2.2}
PYTHON_BIN=${PYTHON_BIN:-/hy-tmp/miniconda3/envs/Wan2.2/bin/python}
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"

cd "${ROOT_DIR}"

tmux new-session -d -s "${SESSION}" \
  "HF_HOME=/hy-tmp/hf-cache TRANSFORMERS_CACHE=/hy-tmp/hf-cache HF_HUB_CACHE=/hy-tmp/hf-cache/hub ${PYTHON_BIN} experiments/adacache_vbench_50step_45f_480p/run_batch.py --convert_model_dtype 2>&1 | tee ${LOG_DIR}/2026-06-16_adacache_vbench_runner.log"

echo "Started tmux session: ${SESSION}"
echo "Attach with: tmux attach -t ${SESSION}"
