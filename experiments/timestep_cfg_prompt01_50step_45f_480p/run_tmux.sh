#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/hy-tmp/miniconda3/envs/Wan2.2/bin/python}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
EXP_ROOT="${EXP_ROOT:-/hy-tmp/wan22_timestep_cfg_prompt01_50step_45f_480p_${STAMP}}"
SESSION="${SESSION:-timestep_cfg_p01_${STAMP}}"
BASELINE_ROOT="${BASELINE_ROOT:-/hy-tmp/wan22_cache_ablation_prompt01_50step_45f_480p_20260609_184625}"
TIMESTEP_THRESHOLDS="${TIMESTEP_THRESHOLDS:-0.10 0.20}"
CFG_THRESHOLDS="${CFG_THRESHOLDS:-0.10 0.20}"

mkdir -p "${EXP_ROOT}"
ln -sfn "${EXP_ROOT}" "${ROOT_DIR}/experiment_results/$(basename "${EXP_ROOT}")"

cat > "${EXP_ROOT}/launch.env" <<EOF
root_dir=${ROOT_DIR}
python_bin=${PYTHON_BIN}
exp_root=${EXP_ROOT}
session=${SESSION}
baseline_root=${BASELINE_ROOT}
timestep_thresholds=${TIMESTEP_THRESHOLDS}
cfg_thresholds=${CFG_THRESHOLDS}
EOF

nvidia-smi > "${EXP_ROOT}/gpu_before_launch.txt" 2>&1 || true

tmux new-session -d -s "${SESSION}" "cd '${ROOT_DIR}' && HF_HOME=/hy-tmp/hf-cache TRANSFORMERS_CACHE=/hy-tmp/hf-cache HF_HUB_CACHE=/hy-tmp/hf-cache/hub '${PYTHON_BIN}' experiments/timestep_cfg_prompt01_50step_45f_480p/run_batch.py --exp_root '${EXP_ROOT}' --baseline_root '${BASELINE_ROOT}' --timestep_thresholds '${TIMESTEP_THRESHOLDS}' --cfg_thresholds '${CFG_THRESHOLDS}' 2>&1 | tee '${EXP_ROOT}/runner.log'"

echo "SESSION=${SESSION}"
echo "EXP_ROOT=${EXP_ROOT}"
