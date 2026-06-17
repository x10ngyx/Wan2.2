#!/usr/bin/env bash
set -euo pipefail

SESSION="${1:-adaptive_seacache_ali12}"
ROOT_DIR="/hy-tmp/work/Wan2.2"
PYTHON="/hy-tmp/miniconda3/envs/Wan2.2/bin/python"
STAMP="$(date +%Y%m%d_%H%M%S)"
EXP_ROOT="/hy-tmp/wan22_adaptive_seacache_ali_prompt12_50step_45f_480p_${STAMP}"

mkdir -p "${EXP_ROOT}/logs"

tmux new-session -d -s "${SESSION}" "cd '${ROOT_DIR}' && \
  export HF_HOME=/hy-tmp/hf-cache TRANSFORMERS_CACHE=/hy-tmp/hf-cache HF_HUB_CACHE=/hy-tmp/hf-cache/hub && \
  '${PYTHON}' experiments/adaptive_seacache_ali_prompt12_50step_45f_480p/run_batch.py \
    --root_dir '${ROOT_DIR}' \
    --python_bin '${PYTHON}' \
    --exp_root '${EXP_ROOT}' \
    --prompt_file '${ROOT_DIR}/test_sets/ali_10/prompts.txt' \
    --prompt_start 0 \
    --prompt_limit 2 \
    --target_psnrs '20 25 30' \
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
