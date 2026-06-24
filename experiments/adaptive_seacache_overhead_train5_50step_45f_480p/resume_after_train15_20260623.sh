#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/hy-tmp/work/Wan2.2"
PYTHON="/hy-tmp/miniconda3/envs/Wan2.2/bin/python"
EXP_ROOT="/hy-tmp/wan22_adaptive_seacache_overhead_train5_50step_45f_480p_20260619_143632"
BASELINE_ROOT="${ROOT_DIR}/experiment_results/openvid_100_seacache_trace_data"
WAIT_FOR_SESSION="adaptive_seacache_train15_test5_resume"
WAIT_LOG="${EXP_ROOT}/logs/overhead_resume_wait_20260623.log"
RUN_LOG="${EXP_ROOT}/logs/runner_resume_20260623.log"

cd "${ROOT_DIR}"
mkdir -p "${EXP_ROOT}/logs"

echo "[$(date '+%F %T')] waiting for ${WAIT_FOR_SESSION}" | tee -a "${WAIT_LOG}"
while tmux has-session -t "${WAIT_FOR_SESSION}" 2>/dev/null; do
  sleep 300
done
echo "[$(date '+%F %T')] starting overhead resume" | tee -a "${WAIT_LOG}"

export HF_HOME=/hy-tmp/hf-cache
export TRANSFORMERS_CACHE=/hy-tmp/hf-cache
export HF_HUB_CACHE=/hy-tmp/hf-cache/hub

"${PYTHON}" experiments/adaptive_seacache_overhead_train5_50step_45f_480p/run_batch.py \
  --root_dir "${ROOT_DIR}" \
  --python_bin "${PYTHON}" \
  --exp_root "${EXP_ROOT}" \
  --baseline_root "${BASELINE_ROOT}" \
  --prompt_jsonl "${ROOT_DIR}/test_sets/openvid_100/prompts.jsonl" \
  --split_json "/hy-tmp/wan22_adaptive_threshold_feature_ablation_hdim16_20260616/temporal_mean/split.json" \
  --random_seed 20260619 \
  --train_prompt_count 5 \
  --test_prompt_count 0 \
  --target_psnrs "20 25 30" \
  --ckpt_dir /hy-tmp/models/Wan2.2-T2V-A14B \
  --size "832*480" \
  --frame_num 45 \
  --sample_steps 50 \
  --sample_solver dpm++ \
  --base_seed 42 \
  --offload_model true \
  --convert_model_dtype \
  --resume_existing \
  2>&1 | tee -a "${RUN_LOG}"
