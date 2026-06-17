#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/hy-tmp/work/Wan2.2}"
PYTHON_BIN="${PYTHON_BIN:-/hy-tmp/miniconda3/envs/Wan2.2/bin/python}"
CKPT_DIR="${CKPT_DIR:-/hy-tmp/models/Wan2.2-T2V-A14B}"
PROMPT_JSONL="${PROMPT_JSONL:-${ROOT_DIR}/test_sets/vbench_every20/prompts.jsonl}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
EXP_ROOT="${EXP_ROOT:-/hy-tmp/wan22_taylorseer_vbench_50step_45f_480p_${STAMP}}"
SESSION="${SESSION:-wan22_taylorseer_vbench}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
ULYSSES_SIZE="${ULYSSES_SIZE:-${NPROC_PER_NODE}}"
PROMPT_START="${PROMPT_START:-0}"
PROMPT_LIMIT="${PROMPT_LIMIT:-0}"
FRESH_THRESHOLD="${FRESH_THRESHOLD:-5}"
MAX_ORDER="${MAX_ORDER:-1}"
FIRST_ENHANCE="${FIRST_ENHANCE:-1}"
SAMPLE_STEPS="${SAMPLE_STEPS:-50}"
FRAME_NUM="${FRAME_NUM:-45}"
SIZE="${SIZE:-832*480}"
BASE_SEED="${BASE_SEED:-42}"
RESUME_EXISTING="${RESUME_EXISTING:-False}"

mkdir -p "${EXP_ROOT}" "${ROOT_DIR}/logs"

resume_arg=()
if [[ "${RESUME_EXISTING}" == "True" || "${RESUME_EXISTING}" == "true" || "${RESUME_EXISTING}" == "1" ]]; then
  resume_arg=(--resume_existing)
fi

CMD=(
  torchrun
  --nproc_per_node="${NPROC_PER_NODE}"
  -m experiments.taylorseer_vbench_50step_45f_480p.run_batch
  --python_bin "${PYTHON_BIN}"
  --ckpt_dir "${CKPT_DIR}"
  --prompt_jsonl "${PROMPT_JSONL}"
  --exp_root "${EXP_ROOT}"
  --task t2v-A14B
  --size "${SIZE}"
  --frame_num "${FRAME_NUM}"
  --sample_steps "${SAMPLE_STEPS}"
  --sample_solver dpm++
  --base_seed "${BASE_SEED}"
  --dit_fsdp
  --t5_fsdp
  --ulysses_size "${ULYSSES_SIZE}"
  --taylorseer_fresh_threshold "${FRESH_THRESHOLD}"
  --taylorseer_max_order "${MAX_ORDER}"
  --taylorseer_first_enhance "${FIRST_ENHANCE}"
  --prompt_start "${PROMPT_START}"
  --prompt_limit "${PROMPT_LIMIT}"
  "${resume_arg[@]}"
)

{
  echo "root_dir=${ROOT_DIR}"
  echo "python_bin=${PYTHON_BIN}"
  echo "ckpt_dir=${CKPT_DIR}"
  echo "prompt_jsonl=${PROMPT_JSONL}"
  echo "exp_root=${EXP_ROOT}"
  echo "session=${SESSION}"
  echo "nproc_per_node=${NPROC_PER_NODE}"
  echo "ulysses_size=${ULYSSES_SIZE}"
  echo "fresh_threshold=${FRESH_THRESHOLD}"
  echo "max_order=${MAX_ORDER}"
  echo "first_enhance=${FIRST_ENHANCE}"
  printf 'command='
  printf '%q ' "${CMD[@]}"
  echo
} | tee "${EXP_ROOT}/launch.env"

tmux new-session -d -s "${SESSION}" \
  "cd '${ROOT_DIR}' && export HF_HOME=/hy-tmp/hf-cache TRANSFORMERS_CACHE=/hy-tmp/hf-cache HF_HUB_CACHE=/hy-tmp/hf-cache/hub && ${CMD[*]} 2>&1 | tee '${EXP_ROOT}/runner.log'"

echo "Launched tmux session ${SESSION}"
echo "Experiment root: ${EXP_ROOT}"
echo "Runner log: ${EXP_ROOT}/runner.log"
