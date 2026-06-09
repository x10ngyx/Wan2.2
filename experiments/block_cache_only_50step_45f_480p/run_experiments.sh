#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/hy-tmp/miniconda3/envs/Wan2.2/bin/python}"
CKPT_DIR="${CKPT_DIR:-/hy-tmp/models/Wan2.2-T2V-A14B}"
PROMPT_FILE="${PROMPT_FILE:-/hy-tmp/work/Wan2.2/prompt.txt}"
FFPROBE_BIN="${FFPROBE_BIN:-/hy-tmp/miniconda3/envs/Wan2.2/bin/ffprobe}"
EXP_ROOT="${EXP_ROOT:-/hy-tmp/wan22_block_cache_only_50step_45f_480p_$(date +%Y%m%d_%H%M%S)}"

TASK="${TASK:-t2v-A14B}"
SIZE="${SIZE:-832*480}"
FRAME_NUM="${FRAME_NUM:-45}"
SAMPLE_STEPS="${SAMPLE_STEPS:-50}"
SAMPLE_SOLVER="${SAMPLE_SOLVER:-dpm++}"
BASE_SEED="${BASE_SEED:-42}"
PROMPT_OFFSET="${PROMPT_OFFSET:-0}"
RESUME_EXISTING="${RESUME_EXISTING:-False}"

BWCACHE_THRESHOLDS="${BWCACHE_THRESHOLDS:-0.05 0.15 0.30}"
BLOCK_GROUP_THRESHOLDS="${BLOCK_GROUP_THRESHOLDS:-0.01 0.03 0.05}"
EXPECTED_BWCACHE_THRESHOLD_COUNT="${EXPECTED_BWCACHE_THRESHOLD_COUNT:-3}"
EXPECTED_BLOCK_GROUP_THRESHOLD_COUNT="${EXPECTED_BLOCK_GROUP_THRESHOLD_COUNT:-3}"

BWCACHE_REUSE_INTERVAL="${BWCACHE_REUSE_INTERVAL:-3}"
BWCACHE_LAST_STEP="${BWCACHE_LAST_STEP:-0.5}"
BWCACHE_METRIC="${BWCACHE_METRIC:-pooled_rel_l1}"

BLOCK_GROUP_SIZE="${BLOCK_GROUP_SIZE:-5}"
BLOCK_GROUP_METRIC="${BLOCK_GROUP_METRIC:-pooled_rel_l1}"
BLOCK_GROUP_START="${BLOCK_GROUP_START:-0.1}"
BLOCK_GROUP_END="${BLOCK_GROUP_END:-0.9}"
BLOCK_GROUP_MAX_REUSE="${BLOCK_GROUP_MAX_REUSE:-3}"
BLOCK_GROUP_EPS="${BLOCK_GROUP_EPS:-1e-6}"

TOOLS_DIR="${ROOT_DIR}/experiments/zeus_timestep_cache_50step_45f_480p"
THIS_DIR="${ROOT_DIR}/experiments/block_cache_only_50step_45f_480p"

COMMON_ARGS=(
  --task "${TASK}"
  --size "${SIZE}"
  --ckpt_dir "${CKPT_DIR}"
  --sample_solver "${SAMPLE_SOLVER}"
  --sample_steps "${SAMPLE_STEPS}"
  --frame_num "${FRAME_NUM}"
  --offload_model True
  --convert_model_dtype
  --timestep_cache none
  --cfg_cache none
)

threshold_label() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/_}"
  printf 'th_%s' "${value}"
}

run_ffprobe() {
  local video="$1"
  local output="$2"
  "${FFPROBE_BIN}" -v error \
    -count_frames \
    -select_streams v:0 \
    -show_entries stream=width,height,nb_frames,nb_read_frames,r_frame_rate,avg_frame_rate,duration \
    -of json "${video}" > "${output}"
}

write_command() {
  local cmd_file="$1"
  shift
  {
    printf 'cd %q\n' "${ROOT_DIR}"
    printf '%q ' "$@"
    printf '\n'
  } > "${cmd_file}"
}

record_time_from_log() {
  local log="$1"
  local time_file="$2"
  local fallback_seconds="$3"
  local inference_elapsed
  inference_elapsed="$(sed -n 's/.*inference_compute_elapsed_seconds=\([0-9.]*\).*/\1/p' "${log}" | tail -n 1)"
  if [[ -n "${inference_elapsed}" ]]; then
    printf 'elapsed_seconds=%s\n' "${inference_elapsed}" > "${time_file}"
  else
    printf 'elapsed_seconds=%s\n' "${fallback_seconds}" > "${time_file}"
  fi
}

run_generate() {
  local method="$1"
  local method_id="$2"
  local prompt_index="$3"
  local prompt="$4"
  local seed="$5"
  local output="$6"
  local log="$7"
  local time_file="$8"
  local cmd_file="$9"
  shift 9
  local args=("$@")

  if [[ "${RESUME_EXISTING}" == "True" && -f "${output}" && -f "${time_file}" && -s "${EXP_ROOT}/ffprobe/${method_id}_prompt_${prompt_index}.json" ]]; then
    echo "Skipping existing ${method_id} prompt ${prompt_index}"
    return 0
  fi

  write_command "${cmd_file}" "${PYTHON_BIN}" "${ROOT_DIR}/generate.py" "${args[@]}"

  echo "Running ${method_id} prompt ${prompt_index} seed ${seed}"
  set +e
  SECONDS=0
  (
    cd "${ROOT_DIR}"
    "${PYTHON_BIN}" "${ROOT_DIR}/generate.py" "${args[@]}"
  ) 2>&1 | tee "${log}"
  local run_status="${PIPESTATUS[0]}"
  local elapsed="${SECONDS}"
  set -e

  record_time_from_log "${log}" "${time_file}" "${elapsed}"
  if [[ "${run_status}" -ne 0 ]]; then
    {
      echo "method=${method}"
      echo "method_id=${method_id}"
      echo "prompt_index=${prompt_index}"
      echo "seed=${seed}"
      echo "status=${run_status}"
      echo "log=${log}"
      echo "command=${cmd_file}"
    } > "${EXP_ROOT}/failed/${method_id}_prompt_${prompt_index}.txt"
    exit "${run_status}"
  fi

  run_ffprobe "${output}" "${EXP_ROOT}/ffprobe/${method_id}_prompt_${prompt_index}.json"
}

run_psnr() {
  local method_id="$1"
  local prompt_index="$2"
  local candidate="$3"
  local psnr_json="$4"
  local psnr_log="$5"

  if [[ "${RESUME_EXISTING}" == "True" && -s "${psnr_json}" ]]; then
    echo "Skipping existing PSNR ${method_id} prompt ${prompt_index}"
    return 0
  fi

  set +e
  "${PYTHON_BIN}" "${TOOLS_DIR}/compute_psnr.py" \
    --reference "${EXP_ROOT}/baseline/prompt_${prompt_index}.mp4" \
    --candidate "${candidate}" \
    --output "${psnr_json}" \
    2>&1 | tee "${psnr_log}"
  local psnr_status="${PIPESTATUS[0]}"
  set -e
  if [[ "${psnr_status}" -ne 0 ]]; then
    {
      echo "method=psnr"
      echo "method_id=${method_id}"
      echo "prompt_index=${prompt_index}"
      echo "status=${psnr_status}"
      echo "log=${psnr_log}"
    } > "${EXP_ROOT}/failed/psnr_${method_id}_prompt_${prompt_index}.txt"
    exit "${psnr_status}"
  fi
}

mkdir -p "${EXP_ROOT}"/{baseline,bwcache,block_group,thresholds,logs,commands,ffprobe,psnr,results,failed}

{
  echo "experiment_root=${EXP_ROOT}"
  echo "root_dir=${ROOT_DIR}"
  echo "python_bin=${PYTHON_BIN}"
  echo "ckpt_dir=${CKPT_DIR}"
  echo "prompt_file=${PROMPT_FILE}"
  echo "ffprobe_bin=${FFPROBE_BIN}"
  echo "task=${TASK}"
  echo "size=${SIZE}"
  echo "frame_num=${FRAME_NUM}"
  echo "sample_steps=${SAMPLE_STEPS}"
  echo "sample_solver=${SAMPLE_SOLVER}"
  echo "base_seed=${BASE_SEED}"
  echo "prompt_offset=${PROMPT_OFFSET}"
  echo "resume_existing=${RESUME_EXISTING}"
  echo "bwcache_thresholds=${BWCACHE_THRESHOLDS}"
  echo "bwcache_reuse_interval=${BWCACHE_REUSE_INTERVAL}"
  echo "bwcache_last_step=${BWCACHE_LAST_STEP}"
  echo "bwcache_metric=${BWCACHE_METRIC}"
  echo "block_group_thresholds=${BLOCK_GROUP_THRESHOLDS}"
  echo "block_group_size=${BLOCK_GROUP_SIZE}"
  echo "block_group_metric=${BLOCK_GROUP_METRIC}"
  echo "block_group_start=${BLOCK_GROUP_START}"
  echo "block_group_end=${BLOCK_GROUP_END}"
  echo "block_group_max_reuse=${BLOCK_GROUP_MAX_REUSE}"
  echo "block_group_eps=${BLOCK_GROUP_EPS}"
} | tee "${EXP_ROOT}/experiment.env"
nvidia-smi > "${EXP_ROOT}/gpu.txt" 2>&1 || true

mapfile -t PROMPTS < <("${PYTHON_BIN}" "${TOOLS_DIR}/read_prompts.py" "${PROMPT_FILE}")
if [[ "${PROMPT_OFFSET}" -lt 0 || "${PROMPT_OFFSET}" -ge "${#PROMPTS[@]}" ]]; then
  echo "PROMPT_OFFSET ${PROMPT_OFFSET} out of range for ${#PROMPTS[@]} prompts" >&2
  exit 1
fi
prompt="${PROMPTS[${PROMPT_OFFSET}]}"
prompt_index="$(printf '%02d' "$((PROMPT_OFFSET + 1))")"
seed="${BASE_SEED}"

read -r -a BWCACHE_THRESHOLD_VALUES <<< "${BWCACHE_THRESHOLDS}"
read -r -a BLOCK_GROUP_THRESHOLD_VALUES <<< "${BLOCK_GROUP_THRESHOLDS}"

if [[ "${#BWCACHE_THRESHOLD_VALUES[@]}" -ne "${EXPECTED_BWCACHE_THRESHOLD_COUNT}" ]]; then
  echo "Expected ${EXPECTED_BWCACHE_THRESHOLD_COUNT} BWCache thresholds, got ${#BWCACHE_THRESHOLD_VALUES[@]}: ${BWCACHE_THRESHOLDS}" >&2
  exit 1
fi
if [[ "${#BLOCK_GROUP_THRESHOLD_VALUES[@]}" -ne "${EXPECTED_BLOCK_GROUP_THRESHOLD_COUNT}" ]]; then
  echo "Expected ${EXPECTED_BLOCK_GROUP_THRESHOLD_COUNT} block-group thresholds, got ${#BLOCK_GROUP_THRESHOLD_VALUES[@]}: ${BLOCK_GROUP_THRESHOLDS}" >&2
  exit 1
fi

for threshold in "${BWCACHE_THRESHOLD_VALUES[@]}"; do
  label="$(threshold_label "${threshold}")"
  mkdir -p "${EXP_ROOT}/bwcache/${label}" "${EXP_ROOT}/psnr/bwcache_${label}"
  {
    echo "method=bwcache"
    echo "threshold=${threshold}"
    echo "reuse_interval=${BWCACHE_REUSE_INTERVAL}"
    echo "last_step=${BWCACHE_LAST_STEP}"
    echo "metric=${BWCACHE_METRIC}"
  } > "${EXP_ROOT}/thresholds/bwcache_${label}.env"
done

for threshold in "${BLOCK_GROUP_THRESHOLD_VALUES[@]}"; do
  label="$(threshold_label "${threshold}")"
  mkdir -p "${EXP_ROOT}/block_group/${label}" "${EXP_ROOT}/psnr/block_group_${label}"
  {
    echo "method=block_group"
    echo "threshold=${threshold}"
    echo "group_size=${BLOCK_GROUP_SIZE}"
    echo "metric=${BLOCK_GROUP_METRIC}"
    echo "start=${BLOCK_GROUP_START}"
    echo "end=${BLOCK_GROUP_END}"
    echo "max_reuse=${BLOCK_GROUP_MAX_REUSE}"
  } > "${EXP_ROOT}/thresholds/block_group_${label}.env"
done

baseline_output="${EXP_ROOT}/baseline/prompt_${prompt_index}.mp4"
baseline_log="${EXP_ROOT}/logs/baseline_prompt_${prompt_index}.log"
baseline_time="${EXP_ROOT}/logs/baseline_prompt_${prompt_index}.time"
baseline_cmd="${EXP_ROOT}/commands/baseline_prompt_${prompt_index}.sh"
baseline_args=(
  "${COMMON_ARGS[@]}"
  --prompt "${prompt}"
  --base_seed "${seed}"
  --save_file "${baseline_output}"
  --block_cache none
)
run_generate "baseline" "baseline" "${prompt_index}" "${prompt}" "${seed}" "${baseline_output}" "${baseline_log}" "${baseline_time}" "${baseline_cmd}" "${baseline_args[@]}"

for threshold in "${BWCACHE_THRESHOLD_VALUES[@]}"; do
  label="$(threshold_label "${threshold}")"
  method_id="bwcache_${label}"
  output="${EXP_ROOT}/bwcache/${label}/prompt_${prompt_index}.mp4"
  log="${EXP_ROOT}/logs/${method_id}_prompt_${prompt_index}.log"
  time_file="${EXP_ROOT}/logs/${method_id}_prompt_${prompt_index}.time"
  cmd_file="${EXP_ROOT}/commands/${method_id}_prompt_${prompt_index}.sh"
  args=(
    "${COMMON_ARGS[@]}"
    --prompt "${prompt}"
    --base_seed "${seed}"
    --save_file "${output}"
    --block_cache bwcache
    --bwcache_thresh "${threshold}"
    --bwcache_reuse_interval "${BWCACHE_REUSE_INTERVAL}"
    --bwcache_last_step "${BWCACHE_LAST_STEP}"
    --bwcache_metric "${BWCACHE_METRIC}"
  )
  run_generate "bwcache" "${method_id}" "${prompt_index}" "${prompt}" "${seed}" "${output}" "${log}" "${time_file}" "${cmd_file}" "${args[@]}"
  run_psnr "${method_id}" "${prompt_index}" "${output}" "${EXP_ROOT}/psnr/${method_id}/prompt_${prompt_index}.json" "${EXP_ROOT}/psnr/${method_id}/prompt_${prompt_index}.log"
done

for threshold in "${BLOCK_GROUP_THRESHOLD_VALUES[@]}"; do
  label="$(threshold_label "${threshold}")"
  method_id="block_group_${label}"
  output="${EXP_ROOT}/block_group/${label}/prompt_${prompt_index}.mp4"
  log="${EXP_ROOT}/logs/${method_id}_prompt_${prompt_index}.log"
  time_file="${EXP_ROOT}/logs/${method_id}_prompt_${prompt_index}.time"
  cmd_file="${EXP_ROOT}/commands/${method_id}_prompt_${prompt_index}.sh"
  args=(
    "${COMMON_ARGS[@]}"
    --prompt "${prompt}"
    --base_seed "${seed}"
    --save_file "${output}"
    --block_cache block-group
    --block_group_size "${BLOCK_GROUP_SIZE}"
    --block_group_threshold "${threshold}"
    --block_group_metric "${BLOCK_GROUP_METRIC}"
    --block_group_start "${BLOCK_GROUP_START}"
    --block_group_end "${BLOCK_GROUP_END}"
    --block_group_max_reuse "${BLOCK_GROUP_MAX_REUSE}"
    --block_group_eps "${BLOCK_GROUP_EPS}"
  )
  run_generate "block_group" "${method_id}" "${prompt_index}" "${prompt}" "${seed}" "${output}" "${log}" "${time_file}" "${cmd_file}" "${args[@]}"
  run_psnr "${method_id}" "${prompt_index}" "${output}" "${EXP_ROOT}/psnr/${method_id}/prompt_${prompt_index}.json" "${EXP_ROOT}/psnr/${method_id}/prompt_${prompt_index}.log"
done

"${PYTHON_BIN}" "${THIS_DIR}/summarize_results.py" \
  --experiment-root "${EXP_ROOT}" \
  --output "${EXP_ROOT}/results/summary.csv"

mkdir -p "${ROOT_DIR}/experiment_results"
ln -sfn "${EXP_ROOT}" "${ROOT_DIR}/experiment_results/$(basename "${EXP_ROOT}")"

echo "Experiment complete: ${EXP_ROOT}"
