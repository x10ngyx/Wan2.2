#!/usr/bin/env bash
set -euo pipefail

cd /hy-tmp/work/Wan2.2

STAMP="$(date +%Y%m%d_%H%M%S)"
EXP_ROOT="/hy-tmp/wan22_adaptive_seacache_mlp_gated_5feature_range_split_compare_50step_45f_480p_${STAMP}"
SESSION="wan22_adaptive_mlp_gated5_split_${STAMP}"
WAIT_FOR_SESSION="${WAIT_FOR_SESSION:-}"

mkdir -p "${EXP_ROOT}/logs"

export HF_HOME=/hy-tmp/hf-cache
export TRANSFORMERS_CACHE=/hy-tmp/hf-cache
export HF_HUB_CACHE=/hy-tmp/hf-cache/hub
export PYTHONUNBUFFERED=1

RUN_CMD="/hy-tmp/miniconda3/envs/Wan2.2/bin/python experiments/adaptive_seacache_mini_dit_split_compare_50step_45f_480p/run_batch.py \
  --exp_root '${EXP_ROOT}' \
  --python_bin /hy-tmp/miniconda3/envs/Wan2.2/bin/python \
  --ffprobe_bin /hy-tmp/miniconda3/envs/Wan2.2/bin/ffprobe \
  --sample_split_json /hy-tmp/wan22_adaptive_threshold_mlp_gated_5feature_range_samplesplit_20260630_035000/split.json \
  --sample_split_model /hy-tmp/wan22_adaptive_threshold_mlp_gated_5feature_range_samplesplit_20260630_035000/best_model_checkpoint.pt \
  --row_split_model /hy-tmp/wan22_adaptive_threshold_mlp_gated_5feature_range_rowsplit_gpu_long100_20260630_035000/best_model_checkpoint.pt \
  --target_psnrs '22 28' \
  --prompt_count 3 \
  --adaptive_min_threshold 0.10 \
  --adaptive_max_threshold 0.80 \
  --resume_existing"

if [[ -n "${WAIT_FOR_SESSION}" ]]; then
  TMUX_CMD="while tmux has-session -t '${WAIT_FOR_SESSION}' 2>/dev/null; do echo \"waiting_for_tmux_session=${WAIT_FOR_SESSION} date=\$(date -Is)\"; sleep 60; done; ${RUN_CMD} 2>&1 | tee '${EXP_ROOT}/logs/runner.log'"
else
  TMUX_CMD="${RUN_CMD} 2>&1 | tee '${EXP_ROOT}/logs/runner.log'"
fi

tmux new-session -d -s "${SESSION}" "${TMUX_CMD}"
ln -sfn "${EXP_ROOT}" "experiment_results/$(basename "${EXP_ROOT}")"

echo "tmux_session=${SESSION}"
echo "exp_root=${EXP_ROOT}"
echo "wait_for_session=${WAIT_FOR_SESSION}"
echo "attach=tmux attach -t ${SESSION}"
echo "log=${EXP_ROOT}/logs/runner.log"
