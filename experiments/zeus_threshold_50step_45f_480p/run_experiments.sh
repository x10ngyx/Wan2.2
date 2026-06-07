#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/hy-tmp/miniconda3/envs/Wan2.2/bin/python}"
CKPT_DIR="${CKPT_DIR:-/hy-tmp/models/Wan2.2-T2V-A14B}"
PROMPT_FILE="${PROMPT_FILE:-/hy-tmp/work/Wan2.2/prompt.txt}"
EXP_ROOT="${EXP_ROOT:-/hy-tmp/wan22_zeus_threshold_50step_45f_480p_$(date +%Y%m%d_%H%M%S)}"

TASK="${TASK:-t2v-A14B}"
SIZE="${SIZE:-832*480}"
FRAME_NUM="${FRAME_NUM:-45}"
SAMPLE_STEPS="${SAMPLE_STEPS:-50}"
SAMPLE_SOLVER="${SAMPLE_SOLVER:-dpm++}"
BASE_SEED="${BASE_SEED:-20260608}"
THRESHOLDS="${THRESHOLDS:-0.03 0.08 0.15 0.30 0.60}"

ZEUS_ACC_START="${ZEUS_ACC_START:-8}"
ZEUS_ACC_END="${ZEUS_ACC_END:-47}"
ZEUS_CACHING_MODE="${ZEUS_CACHING_MODE:-reuse_interp}"
ZEUS_MAX_INTERVAL="${ZEUS_MAX_INTERVAL:-6}"
ZEUS_THRESHOLD_METRIC="${ZEUS_THRESHOLD_METRIC:-rel_l1}"
ZEUS_THRESHOLD_EPS="${ZEUS_THRESHOLD_EPS:-1e-6}"

TOOLS_DIR="${ROOT_DIR}/experiments/zeus_timestep_cache_50step_45f_480p"

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

threshold_label() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/_}"
  printf 'th_%s' "${value}"
}

mkdir -p "${EXP_ROOT}"/{baseline,zeus_threshold,thresholds,logs,commands,ffprobe,psnr,results,failed}

{
  echo "experiment_root=${EXP_ROOT}"
  echo "root_dir=${ROOT_DIR}"
  echo "python_bin=${PYTHON_BIN}"
  echo "ckpt_dir=${CKPT_DIR}"
  echo "prompt_file=${PROMPT_FILE}"
  echo "task=${TASK}"
  echo "size=${SIZE}"
  echo "frame_num=${FRAME_NUM}"
  echo "sample_steps=${SAMPLE_STEPS}"
  echo "sample_solver=${SAMPLE_SOLVER}"
  echo "base_seed=${BASE_SEED}"
  echo "thresholds=${THRESHOLDS}"
  echo "zeus_acc_start=${ZEUS_ACC_START}"
  echo "zeus_acc_end=${ZEUS_ACC_END}"
  echo "zeus_caching_mode=${ZEUS_CACHING_MODE}"
  echo "zeus_max_interval=${ZEUS_MAX_INTERVAL}"
  echo "zeus_threshold_metric=${ZEUS_THRESHOLD_METRIC}"
  echo "zeus_threshold_eps=${ZEUS_THRESHOLD_EPS}"
} | tee "${EXP_ROOT}/experiment.env"
nvidia-smi > "${EXP_ROOT}/gpu.txt" 2>&1 || true

mapfile -t PROMPTS < <("${PYTHON_BIN}" "${TOOLS_DIR}/read_prompts.py" "${PROMPT_FILE}")
read -r -a THRESHOLD_VALUES <<< "${THRESHOLDS}"

if [[ "${#PROMPTS[@]}" -ne 10 ]]; then
  echo "Expected 10 prompts, got ${#PROMPTS[@]} from ${PROMPT_FILE}" >&2
  exit 1
fi

if [[ "${#THRESHOLD_VALUES[@]}" -ne 5 ]]; then
  echo "Expected 5 thresholds, got ${#THRESHOLD_VALUES[@]}: ${THRESHOLDS}" >&2
  exit 1
fi

for threshold in "${THRESHOLD_VALUES[@]}"; do
  label="$(threshold_label "${threshold}")"
  mkdir -p "${EXP_ROOT}/zeus_threshold/${label}" "${EXP_ROOT}/psnr/${label}"
  echo "threshold=${threshold}" > "${EXP_ROOT}/thresholds/${label}.env"
done

run_baseline() {
  local index="$1"
  local prompt="$2"
  local seed="$3"
  local output="${EXP_ROOT}/baseline/prompt_${index}.mp4"
  local log="${EXP_ROOT}/logs/baseline_prompt_${index}.log"
  local time_file="${EXP_ROOT}/logs/baseline_prompt_${index}.time"
  local cmd_file="${EXP_ROOT}/commands/baseline_prompt_${index}.sh"

  local args=("${COMMON_ARGS[@]}" --prompt "${prompt}" --base_seed "${seed}" --save_file "${output}" --timestep_cache none)

  {
    printf 'cd %q\n' "${ROOT_DIR}"
    printf '%q ' "${PYTHON_BIN}" "${ROOT_DIR}/generate.py" "${args[@]}"
    printf '\n'
  } > "${cmd_file}"

  echo "Running baseline prompt ${index} seed ${seed}"
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
      echo "method=baseline"
      echo "prompt_index=${index}"
      echo "seed=${seed}"
      echo "status=${run_status}"
      echo "log=${log}"
      echo "command=${cmd_file}"
    } > "${EXP_ROOT}/failed/baseline_prompt_${index}.txt"
    exit "${run_status}"
  fi

  ffprobe -v error \
    -count_frames \
    -select_streams v:0 \
    -show_entries stream=width,height,nb_frames,nb_read_frames,r_frame_rate,avg_frame_rate,duration \
    -of json "${output}" > "${EXP_ROOT}/ffprobe/baseline_prompt_${index}.json"
}

run_threshold() {
  local threshold="$1"
  local index="$2"
  local prompt="$3"
  local seed="$4"
  local label
  label="$(threshold_label "${threshold}")"
  local method_id="zeus_threshold_${label}"
  local output="${EXP_ROOT}/zeus_threshold/${label}/prompt_${index}.mp4"
  local log="${EXP_ROOT}/logs/${method_id}_prompt_${index}.log"
  local time_file="${EXP_ROOT}/logs/${method_id}_prompt_${index}.time"
  local cmd_file="${EXP_ROOT}/commands/${method_id}_prompt_${index}.sh"

  local args=(
    "${COMMON_ARGS[@]}"
    --prompt "${prompt}"
    --base_seed "${seed}"
    --save_file "${output}"
    --timestep_cache zeus-threshold
    --zeus_acc_start "${ZEUS_ACC_START}"
    --zeus_acc_end "${ZEUS_ACC_END}"
    --zeus_caching_mode "${ZEUS_CACHING_MODE}"
    --zeus_max_interval "${ZEUS_MAX_INTERVAL}"
    --zeus_threshold "${threshold}"
    --zeus_threshold_metric "${ZEUS_THRESHOLD_METRIC}"
    --zeus_threshold_eps "${ZEUS_THRESHOLD_EPS}"
  )

  {
    printf 'cd %q\n' "${ROOT_DIR}"
    printf '%q ' "${PYTHON_BIN}" "${ROOT_DIR}/generate.py" "${args[@]}"
    printf '\n'
  } > "${cmd_file}"

  echo "Running zeus-threshold ${threshold} prompt ${index} seed ${seed}"
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
      echo "method=zeus-threshold"
      echo "threshold=${threshold}"
      echo "threshold_label=${label}"
      echo "prompt_index=${index}"
      echo "seed=${seed}"
      echo "status=${run_status}"
      echo "log=${log}"
      echo "command=${cmd_file}"
    } > "${EXP_ROOT}/failed/${method_id}_prompt_${index}.txt"
    exit "${run_status}"
  fi

  ffprobe -v error \
    -count_frames \
    -select_streams v:0 \
    -show_entries stream=width,height,nb_frames,nb_read_frames,r_frame_rate,avg_frame_rate,duration \
    -of json "${output}" > "${EXP_ROOT}/ffprobe/${method_id}_prompt_${index}.json"

  set +e
  "${PYTHON_BIN}" "${TOOLS_DIR}/compute_psnr.py" \
    --reference "${EXP_ROOT}/baseline/prompt_${index}.mp4" \
    --candidate "${output}" \
    --output "${EXP_ROOT}/psnr/${label}/prompt_${index}.json" \
    2>&1 | tee "${EXP_ROOT}/psnr/${label}/prompt_${index}.log"
  local psnr_status="${PIPESTATUS[0]}"
  set -e
  if [[ "${psnr_status}" -ne 0 ]]; then
    {
      echo "method=psnr"
      echo "threshold=${threshold}"
      echo "threshold_label=${label}"
      echo "prompt_index=${index}"
      echo "status=${psnr_status}"
      echo "log=${EXP_ROOT}/psnr/${label}/prompt_${index}.log"
    } > "${EXP_ROOT}/failed/psnr_${method_id}_prompt_${index}.txt"
    exit "${psnr_status}"
  fi
}

for i in "${!PROMPTS[@]}"; do
  prompt_index="$(printf '%02d' "$((i + 1))")"
  seed="$((BASE_SEED + i))"
  run_baseline "${prompt_index}" "${PROMPTS[$i]}" "${seed}"
  for threshold in "${THRESHOLD_VALUES[@]}"; do
    run_threshold "${threshold}" "${prompt_index}" "${PROMPTS[$i]}" "${seed}"
  done
done

set +e
"${PYTHON_BIN}" "${ROOT_DIR}/experiments/zeus_threshold_50step_45f_480p/summarize_results.py" \
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
