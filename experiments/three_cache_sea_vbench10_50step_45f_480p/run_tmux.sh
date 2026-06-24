#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -x /hy-tmp/env/Wan2.2/bin/python ]]; then
  DEFAULT_PYTHON=/hy-tmp/env/Wan2.2/bin/python
else
  DEFAULT_PYTHON=/hy-tmp/miniconda3/envs/Wan2.2/bin/python
fi
if [[ -x /hy-tmp/env/Wan2.2/bin/ffprobe ]]; then
  DEFAULT_FFPROBE=/hy-tmp/env/Wan2.2/bin/ffprobe
else
  DEFAULT_FFPROBE=/hy-tmp/miniconda3/envs/Wan2.2/bin/ffprobe
fi

PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON}}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
EXP_ROOT="${EXP_ROOT:-/hy-tmp/wan22_three_cache_sea_vbench10_50step_45f_480p_${STAMP}}"
SESSION="${SESSION:-wan22_3sea_vbench10_${STAMP}}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-${CUDA_VISIBLE_DEVICES:-0}}"
PROMPT_PATH="${PROMPT_PATH:-${ROOT_DIR}/test_sets/Vbench10/prompts.jsonl}"
TIMESTEP_THRESHOLDS="${TIMESTEP_THRESHOLDS:-0.05 0.10 0.20 0.50}"
BLOCK_THRESHOLDS="${BLOCK_THRESHOLDS:-0.05 0.10 0.20 0.50}"
CFG_THRESHOLDS="${CFG_THRESHOLDS:-0.05 0.10 0.20 0.50}"
FFPROBE_BIN="${FFPROBE_BIN:-${DEFAULT_FFPROBE}}"
PROMPT_START="${PROMPT_START:-0}"
PROMPT_LIMIT="${PROMPT_LIMIT:-0}"
RESUME_EXISTING="${RESUME_EXISTING:-True}"
COMBO_LIMIT="${COMBO_LIMIT:-0}"

mkdir -p "${EXP_ROOT}" "${ROOT_DIR}/experiment_results"
ln -sfn "${EXP_ROOT}" "${ROOT_DIR}/experiment_results/$(basename "${EXP_ROOT}")"

cat > "${EXP_ROOT}/launch.env" <<EOF
root_dir=${ROOT_DIR}
python_bin=${PYTHON_BIN}
exp_root=${EXP_ROOT}
session=${SESSION}
cuda_visible_devices=${CUDA_VISIBLE_DEVICES_VALUE}
prompt_path=${PROMPT_PATH}
timestep_thresholds=${TIMESTEP_THRESHOLDS}
block_thresholds=${BLOCK_THRESHOLDS}
cfg_thresholds=${CFG_THRESHOLDS}
ffprobe_bin=${FFPROBE_BIN}
prompt_start=${PROMPT_START}
prompt_limit=${PROMPT_LIMIT}
resume_existing=${RESUME_EXISTING}
combo_limit=${COMBO_LIMIT}
timestep_cache=seacache
block_cache=block-group
block_group_metric=sea_full_rel_l1
block_group_decision=accumulated
cfg_cache=sea-threshold
cache_order=CFG outer; branch timestep; block-group on timestep miss
per_prompt_result_dirs=true
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

tmux new-session -d -s "${SESSION}" "cd '${ROOT_DIR}' && export PATH='$(dirname "${PYTHON_BIN}"):$(dirname "${FFPROBE_BIN}"):/hy-tmp/env/Wan2.2/bin:/hy-tmp/miniconda3/envs/Wan2.2/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:'\"\\\$PATH\" && CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES_VALUE}' HF_HOME=/hy-tmp/hf-cache TRANSFORMERS_CACHE=/hy-tmp/hf-cache HF_HUB_CACHE=/hy-tmp/hf-cache/hub '${PYTHON_BIN}' experiments/three_cache_sea_vbench10_50step_45f_480p/run_batch.py --exp_root '${EXP_ROOT}' --prompt_path '${PROMPT_PATH}' --timestep_thresholds '${TIMESTEP_THRESHOLDS}' --block_thresholds '${BLOCK_THRESHOLDS}' --cfg_thresholds '${CFG_THRESHOLDS}' --ffprobe_bin '${FFPROBE_BIN}' --prompt_start '${PROMPT_START}' --prompt_limit '${PROMPT_LIMIT}' ${resume_arg[*]} ${limit_arg[*]} 2>&1 | tee '${EXP_ROOT}/runner.log'"

echo "SESSION=${SESSION}"
echo "EXP_ROOT=${EXP_ROOT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_VALUE}"
echo "Attach: tmux attach -t ${SESSION}"
