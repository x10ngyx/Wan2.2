#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/hy-tmp/miniconda3/envs/Wan2.2/bin/python}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
EXP_ROOT="${EXP_ROOT:-/hy-tmp/wan22_cache_ablation_prompt01_50step_45f_480p_${STAMP}}"
SESSION="${SESSION:-cache_ablation_p01_${STAMP}}"
BASELINE_VIDEO="${BASELINE_VIDEO:-/hy-tmp/wan22_zeus_threshold_reuse_interp_10prompt_5th_20260608_195427/baseline/prompt_01.mp4}"
BASELINE_FFPROBE="${BASELINE_FFPROBE:-/hy-tmp/wan22_zeus_threshold_reuse_interp_10prompt_5th_20260608_195427/ffprobe/baseline_prompt_01.json}"

mkdir -p "${EXP_ROOT}"
ln -sfn "${EXP_ROOT}" "${ROOT_DIR}/experiment_results/$(basename "${EXP_ROOT}")"

cat > "${EXP_ROOT}/launch.env" <<EOF
root_dir=${ROOT_DIR}
python_bin=${PYTHON_BIN}
exp_root=${EXP_ROOT}
session=${SESSION}
baseline_video=${BASELINE_VIDEO}
baseline_ffprobe=${BASELINE_FFPROBE}
EOF

nvidia-smi > "${EXP_ROOT}/gpu_before_launch.txt" 2>&1 || true

tmux new-session -d -s "${SESSION}" "cd '${ROOT_DIR}' && HF_HOME=/hy-tmp/hf-cache TRANSFORMERS_CACHE=/hy-tmp/hf-cache HF_HUB_CACHE=/hy-tmp/hf-cache/hub '${PYTHON_BIN}' experiments/cache_ablation_prompt01_50step_45f_480p/run_ablation.py --exp-root '${EXP_ROOT}' --baseline-video '${BASELINE_VIDEO}' --baseline-ffprobe '${BASELINE_FFPROBE}' 2>&1 | tee '${EXP_ROOT}/runner.log'"

echo "SESSION=${SESSION}"
echo "EXP_ROOT=${EXP_ROOT}"
