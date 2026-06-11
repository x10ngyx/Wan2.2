#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/hy-tmp/miniconda3/envs/Wan2.2/bin/python}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
EXP_ROOT="${EXP_ROOT:-/hy-tmp/wan22_zeus_threshold_openvid100_50step_45f_480p_${STAMP}}"
SESSION="${SESSION:-zeus_threshold_openvid100_${STAMP}}"
DATASET_ZIP="${DATASET_ZIP:-/hy-tmp/openvid_100_wan22_prompts.zip}"
DATASET_MEMBER="${DATASET_MEMBER:-openvid_100_wan22_prompts/dataset_100.jsonl}"
THRESHOLDS="${THRESHOLDS:-0.001 0.003 0.005 0.008 0.010 0.015 0.020 0.030 0.050 0.080}"
PROMPT_START="${PROMPT_START:-0}"
PROMPT_LIMIT="${PROMPT_LIMIT:-0}"
RESUME_EXISTING="${RESUME_EXISTING:-True}"
ZEUS_CACHING_MODE="${ZEUS_CACHING_MODE:-reuse_interp}"
ZEUS_MAX_INTERVAL="${ZEUS_MAX_INTERVAL:-6}"

mkdir -p "${EXP_ROOT}" "${ROOT_DIR}/experiment_results"
ln -sfn "${EXP_ROOT}" "${ROOT_DIR}/experiment_results/$(basename "${EXP_ROOT}")"

cat > "${EXP_ROOT}/launch.env" <<EOF
root_dir=${ROOT_DIR}
python_bin=${PYTHON_BIN}
exp_root=${EXP_ROOT}
session=${SESSION}
dataset_zip=${DATASET_ZIP}
dataset_member=${DATASET_MEMBER}
thresholds=${THRESHOLDS}
prompt_start=${PROMPT_START}
prompt_limit=${PROMPT_LIMIT}
resume_existing=${RESUME_EXISTING}
timestep_cache=zeus-threshold
zeus_caching_mode=${ZEUS_CACHING_MODE}
zeus_max_interval=${ZEUS_MAX_INTERVAL}
block_cache=none
cfg_cache=none
EOF

nvidia-smi > "${EXP_ROOT}/gpu_before_launch.txt" 2>&1 || true

resume_arg=()
if [[ "${RESUME_EXISTING}" == "True" || "${RESUME_EXISTING}" == "true" || "${RESUME_EXISTING}" == "1" ]]; then
  resume_arg=(--resume_existing)
fi

tmux new-session -d -s "${SESSION}" "cd '${ROOT_DIR}' && HF_HOME=/hy-tmp/hf-cache TRANSFORMERS_CACHE=/hy-tmp/hf-cache HF_HUB_CACHE=/hy-tmp/hf-cache/hub '${PYTHON_BIN}' experiments/zeus_threshold_openvid100_50step_45f_480p/run_batch.py --exp_root '${EXP_ROOT}' --dataset_zip '${DATASET_ZIP}' --dataset_member '${DATASET_MEMBER}' --thresholds '${THRESHOLDS}' --prompt_start '${PROMPT_START}' --prompt_limit '${PROMPT_LIMIT}' --zeus_caching_mode '${ZEUS_CACHING_MODE}' --zeus_max_interval '${ZEUS_MAX_INTERVAL}' ${resume_arg[*]} 2>&1 | tee '${EXP_ROOT}/runner.log'"

echo "SESSION=${SESSION}"
echo "EXP_ROOT=${EXP_ROOT}"
