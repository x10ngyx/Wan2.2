#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/hy-tmp/miniconda3/envs/Wan2.2/bin/python}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
EXP_ROOT="${EXP_ROOT:-/hy-tmp/wan22_seacache_50step_45f_480p_${STAMP}}"
SESSION="${SESSION:-seacache_50step_45f_480p_${STAMP}}"
BASELINE_ROOT="${BASELINE_ROOT:-/hy-tmp/wan22_zeus_threshold_reuse_interp_10prompt_5th_20260608_195427}"
THRESHOLDS="${THRESHOLDS:-0.05 0.10 0.20 0.30 0.50}"
PROMPT_START="${PROMPT_START:-0}"
PROMPT_LIMIT="${PROMPT_LIMIT:-1}"
RESUME_EXISTING="${RESUME_EXISTING:-True}"
GENERATE_BASELINE="${GENERATE_BASELINE:-False}"
SEACACHE_USE_RET_STEPS="${SEACACHE_USE_RET_STEPS:-False}"
SEACACHE_POWER_EXP="${SEACACHE_POWER_EXP:-3.0}"
SEACACHE_NORM_MODE="${SEACACHE_NORM_MODE:-mean}"

mkdir -p "${EXP_ROOT}"
ln -sfn "${EXP_ROOT}" "${ROOT_DIR}/experiment_results/$(basename "${EXP_ROOT}")"

cat > "${EXP_ROOT}/launch.env" <<EOF
root_dir=${ROOT_DIR}
python_bin=${PYTHON_BIN}
exp_root=${EXP_ROOT}
session=${SESSION}
baseline_root=${BASELINE_ROOT}
thresholds=${THRESHOLDS}
prompt_start=${PROMPT_START}
prompt_limit=${PROMPT_LIMIT}
resume_existing=${RESUME_EXISTING}
generate_baseline=${GENERATE_BASELINE}
seacache_use_ret_steps=${SEACACHE_USE_RET_STEPS}
seacache_power_exp=${SEACACHE_POWER_EXP}
seacache_norm_mode=${SEACACHE_NORM_MODE}
block_cache=none
cfg_cache=none
EOF

nvidia-smi > "${EXP_ROOT}/gpu_before_launch.txt" 2>&1 || true

resume_arg=()
if [[ "${RESUME_EXISTING}" == "True" || "${RESUME_EXISTING}" == "true" || "${RESUME_EXISTING}" == "1" ]]; then
  resume_arg=(--resume_existing)
fi

baseline_arg=()
if [[ "${GENERATE_BASELINE}" == "True" || "${GENERATE_BASELINE}" == "true" || "${GENERATE_BASELINE}" == "1" ]]; then
  baseline_arg=(--generate_baseline)
fi

ret_steps_arg=()
if [[ "${SEACACHE_USE_RET_STEPS}" == "True" || "${SEACACHE_USE_RET_STEPS}" == "true" || "${SEACACHE_USE_RET_STEPS}" == "1" ]]; then
  ret_steps_arg=(--seacache_use_ret_steps)
fi

tmux new-session -d -s "${SESSION}" "cd '${ROOT_DIR}' && HF_HOME=/hy-tmp/hf-cache TRANSFORMERS_CACHE=/hy-tmp/hf-cache HF_HUB_CACHE=/hy-tmp/hf-cache/hub '${PYTHON_BIN}' experiments/seacache_50step_45f_480p/run_batch.py --exp_root '${EXP_ROOT}' --baseline_root '${BASELINE_ROOT}' --thresholds '${THRESHOLDS}' --prompt_start '${PROMPT_START}' --prompt_limit '${PROMPT_LIMIT}' --seacache_power_exp '${SEACACHE_POWER_EXP}' --seacache_norm_mode '${SEACACHE_NORM_MODE}' ${resume_arg[*]} ${baseline_arg[*]} ${ret_steps_arg[*]} 2>&1 | tee '${EXP_ROOT}/runner.log'"

echo "SESSION=${SESSION}"
echo "EXP_ROOT=${EXP_ROOT}"
