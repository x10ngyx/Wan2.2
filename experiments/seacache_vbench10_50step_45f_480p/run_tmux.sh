#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -x /hy-tmp/env/Wan2.2/bin/python ]]; then
  DEFAULT_PYTHON=/hy-tmp/env/Wan2.2/bin/python
else
  DEFAULT_PYTHON=/hy-tmp/miniconda3/envs/Wan2.2/bin/python
fi

PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON}}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
EXP_ROOT="${EXP_ROOT:-/hy-tmp/wan22_seacache_vbench10_50step_45f_480p_${STAMP}}"
SESSION="${SESSION:-wan22_seacache_vbench10_${STAMP}}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-${CUDA_VISIBLE_DEVICES:-0}}"
PROMPT_PATH="${PROMPT_PATH:-${ROOT_DIR}/test_sets/Vbench10/prompts.jsonl}"
THRESHOLDS="${THRESHOLDS:-0.10 0.15 0.20 0.25 0.30 0.40 0.50 0.60 0.70 0.80}"
FFPROBE_BIN="${FFPROBE_BIN:-/hy-tmp/env/Wan2.2/bin/ffprobe}"
PROMPT_START="${PROMPT_START:-0}"
PROMPT_LIMIT="${PROMPT_LIMIT:-0}"
RESUME_EXISTING="${RESUME_EXISTING:-True}"
SEACACHE_USE_RET_STEPS="${SEACACHE_USE_RET_STEPS:-False}"
SEACACHE_POWER_EXP="${SEACACHE_POWER_EXP:-3.0}"
SEACACHE_NORM_MODE="${SEACACHE_NORM_MODE:-mean}"

mkdir -p "${EXP_ROOT}" "${ROOT_DIR}/experiment_results"
ln -sfn "${EXP_ROOT}" "${ROOT_DIR}/experiment_results/$(basename "${EXP_ROOT}")"

cat > "${EXP_ROOT}/launch.env" <<EOF
root_dir=${ROOT_DIR}
python_bin=${PYTHON_BIN}
exp_root=${EXP_ROOT}
session=${SESSION}
cuda_visible_devices=${CUDA_VISIBLE_DEVICES_VALUE}
prompt_path=${PROMPT_PATH}
thresholds=${THRESHOLDS}
ffprobe_bin=${FFPROBE_BIN}
prompt_start=${PROMPT_START}
prompt_limit=${PROMPT_LIMIT}
resume_existing=${RESUME_EXISTING}
timestep_cache=seacache
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

ret_steps_arg=()
if [[ "${SEACACHE_USE_RET_STEPS}" == "True" || "${SEACACHE_USE_RET_STEPS}" == "true" || "${SEACACHE_USE_RET_STEPS}" == "1" ]]; then
  ret_steps_arg=(--seacache_use_ret_steps)
fi

tmux new-session -d -s "${SESSION}" "cd '${ROOT_DIR}' && export PATH='/hy-tmp/env/Wan2.2/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:'\"\\\$PATH\" && CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES_VALUE}' HF_HOME=/hy-tmp/hf-cache TRANSFORMERS_CACHE=/hy-tmp/hf-cache HF_HUB_CACHE=/hy-tmp/hf-cache/hub '${PYTHON_BIN}' experiments/seacache_vbench10_50step_45f_480p/run_batch.py --exp_root '${EXP_ROOT}' --prompt_path '${PROMPT_PATH}' --thresholds '${THRESHOLDS}' --ffprobe_bin '${FFPROBE_BIN}' --prompt_start '${PROMPT_START}' --prompt_limit '${PROMPT_LIMIT}' --seacache_power_exp '${SEACACHE_POWER_EXP}' --seacache_norm_mode '${SEACACHE_NORM_MODE}' ${resume_arg[*]} ${ret_steps_arg[*]} 2>&1 | tee '${EXP_ROOT}/runner.log'"

echo "SESSION=${SESSION}"
echo "EXP_ROOT=${EXP_ROOT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_VALUE}"
