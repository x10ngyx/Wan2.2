#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/hy-tmp/work/Wan2.2}"
PYTHON_BIN="${PYTHON_BIN:-/hy-tmp/env/Wan2.2-fa2torch28/bin/python}"
CKPT_DIR="${CKPT_DIR:-/hy-tmp/models/Wan2.2-T2V-A14B}"
PROMPT_PATH="${PROMPT_PATH:-${ROOT_DIR}/test_sets/Vbench10/prompts.jsonl}"
FFPROBE_BIN="${FFPROBE_BIN:-/hy-tmp/env/Wan2.2-fa2torch28/bin/ffprobe}"
FFMPEG_BIN="${FFMPEG_BIN:-/hy-tmp/env/Wan2.2-fa2torch28/bin/ffmpeg}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
EXP_ROOT="${EXP_ROOT:-/hy-tmp/wan22_adacache_vbench10_slow_fast_50step_45f_480p_${STAMP}}"
SESSION_PREFIX="${SESSION_PREFIX:-wan22_adacache_vbench10}"
SLOW_HIGH_CODEBOOK_PRESET="${SLOW_HIGH_CODEBOOK_PRESET:-wan22_50_slow}"
SLOW_LOW_CODEBOOK_PRESET="${SLOW_LOW_CODEBOOK_PRESET:-wan22_50_slow}"
FAST_HIGH_CODEBOOK_PRESET="${FAST_HIGH_CODEBOOK_PRESET:-wan22_50_fast}"
FAST_LOW_CODEBOOK_PRESET="${FAST_LOW_CODEBOOK_PRESET:-wan22_50_fast}"
SAMPLE_STEPS="${SAMPLE_STEPS:-50}"
FRAME_NUM="${FRAME_NUM:-45}"
SIZE="${SIZE:-832*480}"
BASE_SEED="${BASE_SEED:-42}"
RESUME_EXISTING="${RESUME_EXISTING:-False}"
METHOD_SUBPROCESS="${METHOD_SUBPROCESS:-True}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
GPU0_START="${GPU0_START:-0}"
GPU0_LIMIT="${GPU0_LIMIT:-5}"
GPU1_START="${GPU1_START:-5}"
GPU1_LIMIT="${GPU1_LIMIT:-5}"

mkdir -p "${EXP_ROOT}" "${ROOT_DIR}/logs"
cd "${ROOT_DIR}"

resume_arg=()
if [[ "${RESUME_EXISTING}" == "True" || "${RESUME_EXISTING}" == "true" || "${RESUME_EXISTING}" == "1" ]]; then
  resume_arg=(--resume_existing)
fi
subprocess_arg=()
if [[ "${METHOD_SUBPROCESS}" == "True" || "${METHOD_SUBPROCESS}" == "true" || "${METHOD_SUBPROCESS}" == "1" ]]; then
  subprocess_arg=(--method_subprocess)
fi

launch_shard() {
  local gpu="$1"
  local shard="$2"
  local start="$3"
  local limit="$4"
  local session="${SESSION_PREFIX}_gpu${gpu}"
  local log="${EXP_ROOT}/runner_gpu${gpu}.log"
  local cmd=(
    env
    CUDA_VISIBLE_DEVICES="${gpu}"
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}"
    HF_HOME=/hy-tmp/hf-cache
    TRANSFORMERS_CACHE=/hy-tmp/hf-cache
    HF_HUB_CACHE=/hy-tmp/hf-cache/hub
    FFMPEG_BIN="${FFMPEG_BIN}"
    FFPROBE_BIN="${FFPROBE_BIN}"
    "${PYTHON_BIN}"
    experiments/adacache_vbench10_slow_fast_50step_45f_480p/run_batch.py
    --python_bin "${PYTHON_BIN}"
    --ckpt_dir "${CKPT_DIR}"
    --prompt_path "${PROMPT_PATH}"
    --exp_root "${EXP_ROOT}"
    --shard_id "${shard}"
    --task t2v-A14B
    --size "${SIZE}"
    --frame_num "${FRAME_NUM}"
    --sample_steps "${SAMPLE_STEPS}"
    --sample_solver dpm++
    --base_seed "${BASE_SEED}"
    --prompt_start "${start}"
    --prompt_limit "${limit}"
    --ffprobe_bin "${FFPROBE_BIN}"
    --slow_high_codebook_preset "${SLOW_HIGH_CODEBOOK_PRESET}"
    --slow_low_codebook_preset "${SLOW_LOW_CODEBOOK_PRESET}"
    --fast_high_codebook_preset "${FAST_HIGH_CODEBOOK_PRESET}"
    --fast_low_codebook_preset "${FAST_LOW_CODEBOOK_PRESET}"
    "${resume_arg[@]}"
    "${subprocess_arg[@]}"
  )
  {
    echo "session=${session}"
    echo "gpu=${gpu}"
    echo "shard=${shard}"
    echo "prompt_start=${start}"
    echo "prompt_limit=${limit}"
    printf 'command='
    printf '%q ' "${cmd[@]}"
    echo
  } | tee "${EXP_ROOT}/launch_gpu${gpu}.env"
  tmux new-session -d -s "${session}" \
    "cd '${ROOT_DIR}' && ${cmd[*]} 2>&1 | tee '${log}'"
}

launch_shard 0 0 "${GPU0_START}" "${GPU0_LIMIT}"
launch_shard 1 1 "${GPU1_START}" "${GPU1_LIMIT}"

echo "Launched:"
echo "  ${SESSION_PREFIX}_gpu0 -> ${EXP_ROOT}/runner_gpu0.log"
echo "  ${SESSION_PREFIX}_gpu1 -> ${EXP_ROOT}/runner_gpu1.log"
echo "Experiment root: ${EXP_ROOT}"
