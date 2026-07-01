#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SESSION="${SESSION:-wan22_seacache_unipc_queue_${STAMP}}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-${CUDA_VISIBLE_DEVICES:-0}}"

ALI_EXP_ROOT="${ALI_EXP_ROOT:-/hy-tmp/wan22_seacache_unipc_ali10_50step_45f_480p_${STAMP}}"
VBENCH_EXP_ROOT="${VBENCH_EXP_ROOT:-/hy-tmp/wan22_seacache_unipc_vbench10_50step_45f_480p_${STAMP}}"

mkdir -p "${ALI_EXP_ROOT}" "${VBENCH_EXP_ROOT}" "${ROOT_DIR}/experiment_results"
ln -sfn "${ALI_EXP_ROOT}" "${ROOT_DIR}/experiment_results/$(basename "${ALI_EXP_ROOT}")"
ln -sfn "${VBENCH_EXP_ROOT}" "${ROOT_DIR}/experiment_results/$(basename "${VBENCH_EXP_ROOT}")"

cat > "${ALI_EXP_ROOT}/queue.env" <<EOF
queue_session=${SESSION}
queue_order=1
paired_vbench_root=${VBENCH_EXP_ROOT}
EOF
cat > "${VBENCH_EXP_ROOT}/queue.env" <<EOF
queue_session=${SESSION}
queue_order=2
paired_ali_root=${ALI_EXP_ROOT}
EOF

nvidia-smi > "${ALI_EXP_ROOT}/gpu_before_queue_launch.txt" 2>&1 || true

tmux new-session -d -s "${SESSION}" "cd '${ROOT_DIR}' && CUDA_VISIBLE_DEVICES_VALUE='${CUDA_VISIBLE_DEVICES_VALUE}' STAMP='${STAMP}' EXP_ROOT='${ALI_EXP_ROOT}' SESSION='${SESSION}_ali10' bash experiments/seacache_unipc_ali10_50step_45f_480p/run_tmux.sh && while tmux has-session -t '${SESSION}_ali10' 2>/dev/null; do sleep 60; done && CUDA_VISIBLE_DEVICES_VALUE='${CUDA_VISIBLE_DEVICES_VALUE}' STAMP='${STAMP}' EXP_ROOT='${VBENCH_EXP_ROOT}' SESSION='${SESSION}_vbench10' bash experiments/seacache_unipc_vbench10_50step_45f_480p/run_tmux.sh && while tmux has-session -t '${SESSION}_vbench10' 2>/dev/null; do sleep 60; done"

echo "QUEUE_SESSION=${SESSION}"
echo "ALI_EXP_ROOT=${ALI_EXP_ROOT}"
echo "VBENCH_EXP_ROOT=${VBENCH_EXP_ROOT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_VALUE}"
