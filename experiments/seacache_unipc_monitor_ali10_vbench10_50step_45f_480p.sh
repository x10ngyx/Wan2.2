#!/usr/bin/env bash
set -euo pipefail

ALI_ROOT="${ALI_ROOT:-/hy-tmp/wan22_seacache_unipc_ali10_50step_45f_480p_20260627_023222}"
VBENCH_ROOT="${VBENCH_ROOT:-/hy-tmp/wan22_seacache_unipc_vbench10_50step_45f_480p_20260627_023222}"
QUEUE_SESSION="${QUEUE_SESSION:-wan22_seacache_unipc_queue_20260627_023222}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-600}"
LOG_PATH="${LOG_PATH:-${ALI_ROOT}/logs/queue_monitor.log}"

mkdir -p "$(dirname "${LOG_PATH}")"

count_files() {
  local root="$1"
  local pattern="$2"
  find "${root}" -type f -path "${pattern}" 2>/dev/null | wc -l
}

latest_tail() {
  local path="$1"
  if [[ -f "${path}" ]]; then
    tail -20 "${path}" | sed 's/^/    /'
  else
    echo "    missing: ${path}"
  fi
}

while true; do
  {
    echo "===== $(date '+%Y-%m-%d %H:%M:%S') ====="
    echo "[tmux]"
    tmux ls 2>/dev/null || true
    echo
    echo "[gpu]"
    nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader 2>&1 || true
    echo
    for root in "${ALI_ROOT}" "${VBENCH_ROOT}"; do
      echo "[root] ${root}"
      echo "baseline_mp4=$(count_files "${root}" "${root}/baseline/*.mp4")"
      echo "candidate_mp4=$(count_files "${root}" "${root}/seacache/*/*.mp4")"
      echo "candidate_time=$(find "${root}/logs" -maxdepth 1 -type f -name 'seacache_*.time' 2>/dev/null | wc -l)"
      echo "psnr_json=$(count_files "${root}" "${root}/psnr/*/*.json")"
      echo "failed_files=$(find "${root}/failed" -maxdepth 1 -type f 2>/dev/null | wc -l)"
      if find "${root}/failed" -maxdepth 1 -type f 2>/dev/null | grep -q .; then
        echo "[failed]"
        find "${root}/failed" -maxdepth 1 -type f -print 2>/dev/null | sort
      fi
      echo "[runner_tail]"
      latest_tail "${root}/runner.log"
      echo
    done
  } >> "${LOG_PATH}" 2>&1

  if ! tmux has-session -t "${QUEUE_SESSION}" 2>/dev/null; then
    echo "===== $(date '+%Y-%m-%d %H:%M:%S') queue session ended =====" >> "${LOG_PATH}"
    exit 0
  fi

  sleep "${INTERVAL_SECONDS}"
done
