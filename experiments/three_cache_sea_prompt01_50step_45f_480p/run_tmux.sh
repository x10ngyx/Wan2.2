#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/hy-tmp/miniconda3/envs/Wan2.2/bin/python}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
EXP_ROOT="${EXP_ROOT:-/hy-tmp/wan22_three_cache_sea_prompt01_50step_45f_480p_${STAMP}}"
SESSION="${SESSION:-three_cache_sea_p01_${STAMP}}"
BASELINE_ROOT="${BASELINE_ROOT:-/hy-tmp/wan22_cache_ablation_prompt01_50step_45f_480p_20260609_184625}"
TIMESTEP_THRESHOLDS="${TIMESTEP_THRESHOLDS:-0.05 0.10 0.20 0.40 1.00}"
BLOCK_THRESHOLDS="${BLOCK_THRESHOLDS:-0.05 0.10 0.20 0.40 1.00}"
CFG_THRESHOLDS="${CFG_THRESHOLDS:-0.05 0.10 0.20 0.40 1.00}"
RESUME_EXISTING="${RESUME_EXISTING:-True}"
COMBO_LIMIT="${COMBO_LIMIT:-0}"

mkdir -p "${EXP_ROOT}"
ln -sfn "${EXP_ROOT}" "${ROOT_DIR}/experiment_results/$(basename "${EXP_ROOT}")"

cat > "${EXP_ROOT}/launch.env" <<EOF
root_dir=${ROOT_DIR}
python_bin=${PYTHON_BIN}
exp_root=${EXP_ROOT}
session=${SESSION}
baseline_root=${BASELINE_ROOT}
timestep_thresholds=${TIMESTEP_THRESHOLDS}
block_thresholds=${BLOCK_THRESHOLDS}
cfg_thresholds=${CFG_THRESHOLDS}
resume_existing=${RESUME_EXISTING}
combo_limit=${COMBO_LIMIT}
timestep_cache=seacache
block_cache=block-group
block_group_metric=sea_full_rel_l1
block_group_decision=accumulated
cfg_cache=sea-threshold
cache_order=CFG outer; branch timestep; block-group on timestep miss
EOF

nvidia-smi > "${EXP_ROOT}/gpu_before_launch.txt" 2>&1 || true

resume_arg=()
if [[ "${RESUME_EXISTING}" == "True" || "${RESUME_EXISTING}" == "true" || "${RESUME_EXISTING}" == "1" ]]; then
  resume_arg=(--resume_existing)
fi

limit_arg=()
if [[ "${COMBO_LIMIT}" != "0" ]]; then
  limit_arg=(--combo_limit "${COMBO_LIMIT}")
fi

tmux new-session -d -s "${SESSION}" "cd '${ROOT_DIR}' && HF_HOME=/hy-tmp/hf-cache TRANSFORMERS_CACHE=/hy-tmp/hf-cache HF_HUB_CACHE=/hy-tmp/hf-cache/hub '${PYTHON_BIN}' experiments/three_cache_sea_prompt01_50step_45f_480p/run_batch.py --exp_root '${EXP_ROOT}' --baseline_root '${BASELINE_ROOT}' --timestep_thresholds '${TIMESTEP_THRESHOLDS}' --block_thresholds '${BLOCK_THRESHOLDS}' --cfg_thresholds '${CFG_THRESHOLDS}' ${resume_arg[*]} ${limit_arg[*]} 2>&1 | tee '${EXP_ROOT}/runner.log'"

echo "SESSION=${SESSION}"
echo "EXP_ROOT=${EXP_ROOT}"
