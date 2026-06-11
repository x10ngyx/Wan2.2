#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/hy-tmp/miniconda3/envs/Wan2.2/bin/python}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
EXP_ROOT="${EXP_ROOT:-/hy-tmp/wan22_three_cache_threshold_grid_prompt01_50step_45f_480p_${STAMP}}"
SESSION="${SESSION:-three_cache_grid_p01_${STAMP}}"
BASELINE_VIDEO="${BASELINE_VIDEO:-/hy-tmp/wan22_zeus_threshold_reuse_interp_10prompt_5th_20260608_195427/baseline/prompt_01.mp4}"
BASELINE_FFPROBE="${BASELINE_FFPROBE:-/hy-tmp/wan22_zeus_threshold_reuse_interp_10prompt_5th_20260608_195427/ffprobe/baseline_prompt_01.json}"
PROMPT_INDEX="${PROMPT_INDEX:-1}"
TIMESTEP_THRESHOLDS="${TIMESTEP_THRESHOLDS:-0.001 0.005 0.02 0.60}"
BLOCK_THRESHOLDS="${BLOCK_THRESHOLDS:-0.001 0.015 0.03 1.00}"
CFG_THRESHOLDS="${CFG_THRESHOLDS:-0.001 0.02 0.03 1.00}"
RESUME_EXISTING="${RESUME_EXISTING:-True}"
COMBO_LIMIT="${COMBO_LIMIT:-0}"

mkdir -p "${EXP_ROOT}"
ln -sfn "${EXP_ROOT}" "${ROOT_DIR}/experiment_results/$(basename "${EXP_ROOT}")"

cat > "${EXP_ROOT}/launch.env" <<EOF
root_dir=${ROOT_DIR}
python_bin=${PYTHON_BIN}
exp_root=${EXP_ROOT}
session=${SESSION}
prompt_index=${PROMPT_INDEX}
baseline_video=${BASELINE_VIDEO}
baseline_ffprobe=${BASELINE_FFPROBE}
timestep_thresholds=${TIMESTEP_THRESHOLDS}
block_thresholds=${BLOCK_THRESHOLDS}
cfg_thresholds=${CFG_THRESHOLDS}
resume_existing=${RESUME_EXISTING}
combo_limit=${COMBO_LIMIT}
cfg_force_uncond_recompute_on_miss=False
EOF

nvidia-smi > "${EXP_ROOT}/gpu_before_launch.txt" 2>&1 || true

resume_arg=()
if [[ "${RESUME_EXISTING}" == "True" || "${RESUME_EXISTING}" == "true" || "${RESUME_EXISTING}" == "1" ]]; then
  resume_arg=(--resume-existing)
fi

limit_arg=()
if [[ "${COMBO_LIMIT}" != "0" ]]; then
  limit_arg=(--combo-limit "${COMBO_LIMIT}")
fi

tmux new-session -d -s "${SESSION}" "cd '${ROOT_DIR}' && HF_HOME=/hy-tmp/hf-cache TRANSFORMERS_CACHE=/hy-tmp/hf-cache HF_HUB_CACHE=/hy-tmp/hf-cache/hub '${PYTHON_BIN}' experiments/three_cache_threshold_grid_50step_45f_480p/run_grid.py --exp-root '${EXP_ROOT}' --prompt-index '${PROMPT_INDEX}' --baseline-video '${BASELINE_VIDEO}' --baseline-ffprobe '${BASELINE_FFPROBE}' --timestep-thresholds '${TIMESTEP_THRESHOLDS}' --block-thresholds '${BLOCK_THRESHOLDS}' --cfg-thresholds '${CFG_THRESHOLDS}' ${resume_arg[*]} ${limit_arg[*]} 2>&1 | tee '${EXP_ROOT}/runner.log'"

echo "SESSION=${SESSION}"
echo "EXP_ROOT=${EXP_ROOT}"
