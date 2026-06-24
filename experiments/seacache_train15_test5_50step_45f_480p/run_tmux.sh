#!/usr/bin/env bash
set -euo pipefail

SESSION="${1:-seacache_train15_test5}"
ROOT_DIR="/hy-tmp/work/Wan2.2"
PYTHON="/hy-tmp/miniconda3/envs/Wan2.2/bin/python"
STAMP="$(date +%Y%m%d_%H%M%S)"
EXP_ROOT="/hy-tmp/wan22_seacache_train15_test5_50step_45f_480p_${STAMP}"
BASELINE_ROOT="${ROOT_DIR}/experiment_results/openvid_100_seacache_trace_data"

mkdir -p "${EXP_ROOT}/logs"

tmux new-session -d -s "${SESSION}" "cd '${ROOT_DIR}' && \
  export HF_HOME=/hy-tmp/hf-cache TRANSFORMERS_CACHE=/hy-tmp/hf-cache HF_HUB_CACHE=/hy-tmp/hf-cache/hub && \
  '${PYTHON}' experiments/seacache_train15_test5_50step_45f_480p/run_batch.py \
    --root_dir '${ROOT_DIR}' \
    --python_bin '${PYTHON}' \
    --exp_root '${EXP_ROOT}' \
    --baseline_root '${BASELINE_ROOT}' \
    --prompt_jsonl '${ROOT_DIR}/test_sets/openvid_100/prompts.jsonl' \
    --split_json '/hy-tmp/wan22_adaptive_threshold_feature_ablation_hdim16_20260616/temporal_mean/split.json' \
    --random_seed 20260619 \
    --train_prompt_count 15 \
    --test_prompt_count 5 \
    --thresholds '0.1 0.2 0.4 0.6' \
    --ckpt_dir /hy-tmp/models/Wan2.2-T2V-A14B \
    --size '832*480' \
    --frame_num 45 \
    --sample_steps 50 \
    --sample_solver dpm++ \
    --base_seed 42 \
    --offload_model true \
    --convert_model_dtype \
    --resume_existing \
    2>&1 | tee '${EXP_ROOT}/logs/runner.log'"

echo "tmux session: ${SESSION}"
echo "experiment root: ${EXP_ROOT}"
