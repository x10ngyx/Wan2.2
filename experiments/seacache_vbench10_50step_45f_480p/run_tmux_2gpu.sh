#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
PARENT_ROOT="${PARENT_ROOT:-/hy-tmp/wan22_seacache_vbench10_50step_45f_480p_${STAMP}}"
PROMPT_PATH="${PROMPT_PATH:-${ROOT_DIR}/test_sets/Vbench10/prompts.jsonl}"
THRESHOLDS="${THRESHOLDS:-0.10 0.15 0.20 0.25 0.30 0.40 0.50 0.60 0.70 0.80}"
FFPROBE_BIN="${FFPROBE_BIN:-/hy-tmp/env/Wan2.2/bin/ffprobe}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
SESSION_PREFIX="${SESSION_PREFIX:-wan22_seacache_vbench10_${STAMP}}"

mkdir -p "${PARENT_ROOT}" "${ROOT_DIR}/experiment_results"
ln -sfn "${PARENT_ROOT}" "${ROOT_DIR}/experiment_results/$(basename "${PARENT_ROOT}")"

cat > "${PARENT_ROOT}/launch_2gpu.env" <<EOF
root_dir=${ROOT_DIR}
parent_root=${PARENT_ROOT}
prompt_path=${PROMPT_PATH}
thresholds=${THRESHOLDS}
ffprobe_bin=${FFPROBE_BIN}
gpu0=${GPU0}
gpu1=${GPU1}
session_prefix=${SESSION_PREFIX}
shard0_prompt_start=0
shard0_prompt_limit=5
shard1_prompt_start=5
shard1_prompt_limit=5
timestep_cache=seacache
block_cache=none
cfg_cache=none
EOF

CUDA_VISIBLE_DEVICES_VALUE="${GPU0}" \
EXP_ROOT="${PARENT_ROOT}/shard_gpu${GPU0}_p000_004" \
SESSION="${SESSION_PREFIX}_gpu${GPU0}_p000_004" \
PROMPT_PATH="${PROMPT_PATH}" \
THRESHOLDS="${THRESHOLDS}" \
FFPROBE_BIN="${FFPROBE_BIN}" \
PROMPT_START=0 \
PROMPT_LIMIT=5 \
bash "${ROOT_DIR}/experiments/seacache_vbench10_50step_45f_480p/run_tmux.sh"

CUDA_VISIBLE_DEVICES_VALUE="${GPU1}" \
EXP_ROOT="${PARENT_ROOT}/shard_gpu${GPU1}_p005_009" \
SESSION="${SESSION_PREFIX}_gpu${GPU1}_p005_009" \
PROMPT_PATH="${PROMPT_PATH}" \
THRESHOLDS="${THRESHOLDS}" \
FFPROBE_BIN="${FFPROBE_BIN}" \
PROMPT_START=5 \
PROMPT_LIMIT=5 \
bash "${ROOT_DIR}/experiments/seacache_vbench10_50step_45f_480p/run_tmux.sh"

echo "PARENT_ROOT=${PARENT_ROOT}"
echo "GPU0_SESSION=${SESSION_PREFIX}_gpu${GPU0}_p000_004"
echo "GPU1_SESSION=${SESSION_PREFIX}_gpu${GPU1}_p005_009"
echo "Merge after both shards finish:"
echo "  python experiments/seacache_vbench10_50step_45f_480p/merge_shards.py --parent-root '${PARENT_ROOT}'"
