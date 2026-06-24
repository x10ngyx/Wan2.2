#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -x /hy-tmp/miniconda3/envs/Wan2.2/bin/python ]]; then
  DEFAULT_PYTHON=/hy-tmp/miniconda3/envs/Wan2.2/bin/python
else
  DEFAULT_PYTHON=/hy-tmp/env/Wan2.2/bin/python
fi

PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON}}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
EXP_ROOT="${EXP_ROOT:-/hy-tmp/wan22_zeus_vbench10_50step_45f_480p_${STAMP}}"
SESSION="${SESSION:-wan22_zeus_vbench10_${STAMP}}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-${CUDA_VISIBLE_DEVICES:-0}}"
PROMPT_PATH="${PROMPT_PATH:-${ROOT_DIR}/test_sets/Vbench10/prompts.jsonl}"
THRESHOLDS="${THRESHOLDS:-0.005 0.02 0.08 0.20 0.60}"
BASELINE_REUSE_ROOT="${BASELINE_REUSE_ROOT:-${ROOT_DIR}/experiment_results/wan22_vbench10_three_cache_full_merge_and_timestep_only_full_20260623}"
FFPROBE_BIN="${FFPROBE_BIN:-/hy-tmp/miniconda3/envs/Wan2.2/bin/ffprobe}"
PROMPT_START="${PROMPT_START:-0}"
PROMPT_LIMIT="${PROMPT_LIMIT:-0}"
RESUME_EXISTING="${RESUME_EXISTING:-True}"

ZEUS_ACC_START="${ZEUS_ACC_START:-8}"
ZEUS_ACC_END="${ZEUS_ACC_END:-47}"
ZEUS_DENOMINATOR="${ZEUS_DENOMINATOR:-3}"
ZEUS_MODULAR="${ZEUS_MODULAR:-0 1}"
ZEUS_CACHING_MODE="${ZEUS_CACHING_MODE:-reuse_interp}"
ZEUS_MAX_INTERVAL="${ZEUS_MAX_INTERVAL:-6}"
ZEUS_LAGRANGE_TERM="${ZEUS_LAGRANGE_TERM:-4}"
ZEUS_LAGRANGE_INT="${ZEUS_LAGRANGE_INT:-4}"
ZEUS_LAGRANGE_STEP="${ZEUS_LAGRANGE_STEP:-24}"

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
baseline_reuse_root=${BASELINE_REUSE_ROOT}
ffprobe_bin=${FFPROBE_BIN}
prompt_start=${PROMPT_START}
prompt_limit=${PROMPT_LIMIT}
resume_existing=${RESUME_EXISTING}
timestep_cache_methods=zeus,zeus-threshold
block_cache=none
cfg_cache=none
zeus_acc_start=${ZEUS_ACC_START}
zeus_acc_end=${ZEUS_ACC_END}
zeus_denominator=${ZEUS_DENOMINATOR}
zeus_modular=${ZEUS_MODULAR}
zeus_caching_mode=${ZEUS_CACHING_MODE}
zeus_max_interval=${ZEUS_MAX_INTERVAL}
zeus_lagrange_term=${ZEUS_LAGRANGE_TERM}
zeus_lagrange_int=${ZEUS_LAGRANGE_INT}
zeus_lagrange_step=${ZEUS_LAGRANGE_STEP}
EOF

nvidia-smi > "${EXP_ROOT}/gpu_before_launch.txt" 2>&1 || true

resume_arg=()
if [[ "${RESUME_EXISTING}" == "True" || "${RESUME_EXISTING}" == "true" || "${RESUME_EXISTING}" == "1" ]]; then
  resume_arg=(--resume_existing)
fi

tmux new-session -d -s "${SESSION}" "cd '${ROOT_DIR}' && export PATH='/hy-tmp/miniconda3/envs/Wan2.2/bin:/hy-tmp/env/Wan2.2/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:'\"\\\$PATH\" && CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES_VALUE}' HF_HOME=/hy-tmp/hf-cache TRANSFORMERS_CACHE=/hy-tmp/hf-cache HF_HUB_CACHE=/hy-tmp/hf-cache/hub '${PYTHON_BIN}' experiments/zeus_vbench10_50step_45f_480p/run_batch.py --exp_root '${EXP_ROOT}' --prompt_path '${PROMPT_PATH}' --thresholds '${THRESHOLDS}' --baseline_reuse_root '${BASELINE_REUSE_ROOT}' --ffprobe_bin '${FFPROBE_BIN}' --prompt_start '${PROMPT_START}' --prompt_limit '${PROMPT_LIMIT}' --zeus_acc_start '${ZEUS_ACC_START}' --zeus_acc_end '${ZEUS_ACC_END}' --zeus_denominator '${ZEUS_DENOMINATOR}' --zeus_modular ${ZEUS_MODULAR} --zeus_caching_mode '${ZEUS_CACHING_MODE}' --zeus_max_interval '${ZEUS_MAX_INTERVAL}' --zeus_lagrange_term '${ZEUS_LAGRANGE_TERM}' --zeus_lagrange_int '${ZEUS_LAGRANGE_INT}' --zeus_lagrange_step '${ZEUS_LAGRANGE_STEP}' ${resume_arg[*]} 2>&1 | tee '${EXP_ROOT}/runner.log'"

echo "SESSION=${SESSION}"
echo "EXP_ROOT=${EXP_ROOT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_VALUE}"
