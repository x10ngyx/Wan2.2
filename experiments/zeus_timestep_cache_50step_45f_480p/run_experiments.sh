#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/hy-tmp/miniconda3/envs/Wan2.2/bin/python}"
CKPT_DIR="${CKPT_DIR:-/hy-tmp/models/Wan2.2-T2V-A14B}"
PROMPT_FILE="${PROMPT_FILE:-/hy-tmp/work/Wan2.2/prompt.txt}"
EXP_ROOT="${EXP_ROOT:-/hy-tmp/wan22_zeus_timestep_cache_50step_45f_480p_$(date +%Y%m%d_%H%M%S)}"

TASK="${TASK:-t2v-A14B}"
SIZE="${SIZE:-832*480}"
FRAME_NUM="${FRAME_NUM:-45}"
SAMPLE_STEPS="${SAMPLE_STEPS:-50}"
SAMPLE_SOLVER="${SAMPLE_SOLVER:-dpm++}"
BASE_SEED="${BASE_SEED:-20260608}"

ZEUS_ACC_START="${ZEUS_ACC_START:-8}"
ZEUS_ACC_END="${ZEUS_ACC_END:-47}"
ZEUS_DENOMINATOR="${ZEUS_DENOMINATOR:-3}"
ZEUS_MODULAR="${ZEUS_MODULAR:-(0,1)}"
ZEUS_CACHING_MODE="${ZEUS_CACHING_MODE:-reuse_interp}"
ZEUS_MAX_INTERVAL="${ZEUS_MAX_INTERVAL:-6}"
ZEUS_LAGRANGE_TERM="${ZEUS_LAGRANGE_TERM:-4}"
ZEUS_LAGRANGE_INT="${ZEUS_LAGRANGE_INT:-4}"
ZEUS_LAGRANGE_STEP="${ZEUS_LAGRANGE_STEP:-24}"

COMMON_ARGS=(
  --task "${TASK}"
  --size "${SIZE}"
  --ckpt_dir "${CKPT_DIR}"
  --sample_solver "${SAMPLE_SOLVER}"
  --sample_steps "${SAMPLE_STEPS}"
  --frame_num "${FRAME_NUM}"
  --offload_model True
  --convert_model_dtype
  --block_cache none
  --cfg_cache none
)

mkdir -p "${EXP_ROOT}"/{baseline,zeus,logs,commands,ffprobe,psnr,results,failed}

echo "experiment_root=${EXP_ROOT}" | tee "${EXP_ROOT}/experiment.env"
echo "root_dir=${ROOT_DIR}" | tee -a "${EXP_ROOT}/experiment.env"
echo "python_bin=${PYTHON_BIN}" | tee -a "${EXP_ROOT}/experiment.env"
echo "ckpt_dir=${CKPT_DIR}" | tee -a "${EXP_ROOT}/experiment.env"
echo "prompt_file=${PROMPT_FILE}" | tee -a "${EXP_ROOT}/experiment.env"
echo "task=${TASK}" | tee -a "${EXP_ROOT}/experiment.env"
echo "size=${SIZE}" | tee -a "${EXP_ROOT}/experiment.env"
echo "frame_num=${FRAME_NUM}" | tee -a "${EXP_ROOT}/experiment.env"
echo "sample_steps=${SAMPLE_STEPS}" | tee -a "${EXP_ROOT}/experiment.env"
echo "sample_solver=${SAMPLE_SOLVER}" | tee -a "${EXP_ROOT}/experiment.env"
echo "base_seed=${BASE_SEED}" | tee -a "${EXP_ROOT}/experiment.env"
nvidia-smi > "${EXP_ROOT}/gpu.txt" 2>&1 || true

mapfile -t PROMPTS < <("${PYTHON_BIN}" "${ROOT_DIR}/experiments/zeus_timestep_cache_50step_45f_480p/read_prompts.py" "${PROMPT_FILE}")

if [[ "${#PROMPTS[@]}" -ne 10 ]]; then
  echo "Expected 10 prompts, got ${#PROMPTS[@]} from ${PROMPT_FILE}" >&2
  exit 1
fi

run_one() {
  local method="$1"
  local index="$2"
  local prompt="$3"
  local seed="$4"
  local output="${EXP_ROOT}/${method}/prompt_${index}.mp4"
  local log="${EXP_ROOT}/logs/${method}_prompt_${index}.log"
  local time_file="${EXP_ROOT}/logs/${method}_prompt_${index}.time"
  local cmd_file="${EXP_ROOT}/commands/${method}_prompt_${index}.sh"

  local args=("${COMMON_ARGS[@]}" --prompt "${prompt}" --base_seed "${seed}" --save_file "${output}")
  if [[ "${method}" == "zeus" ]]; then
    args+=(
      --timestep_cache zeus
      --zeus_acc_start "${ZEUS_ACC_START}"
      --zeus_acc_end "${ZEUS_ACC_END}"
      --zeus_denominator "${ZEUS_DENOMINATOR}"
      --zeus_modular "${ZEUS_MODULAR}"
      --zeus_caching_mode "${ZEUS_CACHING_MODE}"
      --zeus_max_interval "${ZEUS_MAX_INTERVAL}"
      --zeus_lagrange_term "${ZEUS_LAGRANGE_TERM}"
      --zeus_lagrange_int "${ZEUS_LAGRANGE_INT}"
      --zeus_lagrange_step "${ZEUS_LAGRANGE_STEP}"
    )
  else
    args+=(--timestep_cache none)
  fi

  {
    printf 'cd %q\n' "${ROOT_DIR}"
    printf '%q ' "${PYTHON_BIN}" "${ROOT_DIR}/generate.py" "${args[@]}"
    printf '\n'
  } > "${cmd_file}"

  echo "Running ${method} prompt ${index} seed ${seed}"
  set +e
  (
    cd "${ROOT_DIR}"
    /usr/bin/time -f 'elapsed_seconds=%e' -o "${time_file}" \
      "${PYTHON_BIN}" "${ROOT_DIR}/generate.py" "${args[@]}"
  ) 2>&1 | tee "${log}"
  local run_status="${PIPESTATUS[0]}"
  set -e
  if [[ "${run_status}" -ne 0 ]]; then
    {
      echo "method=${method}"
      echo "prompt_index=${index}"
      echo "seed=${seed}"
      echo "status=${run_status}"
      echo "log=${log}"
      echo "command=${cmd_file}"
    } > "${EXP_ROOT}/failed/${method}_prompt_${index}.txt"
    exit "${run_status}"
  fi

  ffprobe -v error \
    -count_frames \
    -select_streams v:0 \
    -show_entries stream=width,height,nb_frames,nb_read_frames,r_frame_rate,avg_frame_rate,duration \
    -of json "${output}" > "${EXP_ROOT}/ffprobe/${method}_prompt_${index}.json"
}

for i in "${!PROMPTS[@]}"; do
  prompt_index="$(printf '%02d' "$((i + 1))")"
  seed="$((BASE_SEED + i))"
  run_one baseline "${prompt_index}" "${PROMPTS[$i]}" "${seed}"
  run_one zeus "${prompt_index}" "${PROMPTS[$i]}" "${seed}"

  set +e
  "${PYTHON_BIN}" "${ROOT_DIR}/experiments/zeus_timestep_cache_50step_45f_480p/compute_psnr.py" \
    --reference "${EXP_ROOT}/baseline/prompt_${prompt_index}.mp4" \
    --candidate "${EXP_ROOT}/zeus/prompt_${prompt_index}.mp4" \
    --output "${EXP_ROOT}/psnr/prompt_${prompt_index}.json" \
    2>&1 | tee "${EXP_ROOT}/psnr/prompt_${prompt_index}.log"
  psnr_status="${PIPESTATUS[0]}"
  set -e
  if [[ "${psnr_status}" -ne 0 ]]; then
    {
      echo "method=psnr"
      echo "prompt_index=${prompt_index}"
      echo "status=${psnr_status}"
      echo "log=${EXP_ROOT}/psnr/prompt_${prompt_index}.log"
    } > "${EXP_ROOT}/failed/psnr_prompt_${prompt_index}.txt"
    exit "${psnr_status}"
  fi
done

set +e
"${PYTHON_BIN}" "${ROOT_DIR}/experiments/zeus_timestep_cache_50step_45f_480p/summarize_results.py" \
  --experiment-root "${EXP_ROOT}" \
  --output "${EXP_ROOT}/results/summary.csv" \
  2>&1 | tee "${EXP_ROOT}/results/summary.log"
summary_status="${PIPESTATUS[0]}"
set -e
if [[ "${summary_status}" -ne 0 ]]; then
  {
    echo "method=summary"
    echo "status=${summary_status}"
    echo "log=${EXP_ROOT}/results/summary.log"
  } > "${EXP_ROOT}/failed/summary.txt"
  exit "${summary_status}"
fi

echo "Completed experiment: ${EXP_ROOT}"
