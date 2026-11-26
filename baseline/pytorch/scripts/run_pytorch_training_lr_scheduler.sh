#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
GPT2_INPUT_BIN="${GPT2_INPUT_BIN:-${WORKSPACE_ROOT}/InfiniTrain/data/gpt2/tiny_shakespeare_train.bin}"
LLAMA3_INPUT_BIN="${LLAMA3_INPUT_BIN:-${WORKSPACE_ROOT}/InfiniTrain/data/llama3/tiny_shakespeare_train.bin}"
LLAMA3_LLMC_FILEPATH="${LLAMA3_LLMC_FILEPATH:-${WORKSPACE_ROOT}/InfiniTrain/data/llama3/llama3.2_1B_fp32.bin}"
GPT2_ARGS=(
  --input_bin "${GPT2_INPUT_BIN}"
)
LLAMA_ARGS=(
  --input_bin "${LLAMA3_INPUT_BIN}"
  --llmc_filepath "${LLAMA3_LLMC_FILEPATH}"
)

### Set HF-related environments correctly if needed
# export HF_HOME=/data1/shared/InfiniTrain-dev/env/HuggingFace
# export HF_HUB_CACHE=/data1/shared/InfiniTrain-dev/env/HuggingFace/hub
# export HF_HUB_OFFLINE=1

# This script is used to compare loss only.
# The loss performance of each case should be mathematically equivalent to the following setting with 8-card DDP.
BASE_ARGS=(
  --batch_size 10
  --total_batch_size 5120
  --num_iterations 10
)

TRAINING_SCRIPTS=(
  train_gpt2.py
  train_llama3.2_1B.py
)

run_case() {
  local case_id="$1"
  shift

  for training_script in "${TRAINING_SCRIPTS[@]}"; do
    echo "=========================================="
    echo "Running: ${training_script} / ${case_id}"
    echo "=========================================="
    if [[ "${training_script}" == train_llama3.2_1B.py ]]; then
      torchrun --nproc_per_node=8 "${training_script}" \
        "${BASE_ARGS[@]}" "${LLAMA_ARGS[@]}" "$@"
    else
      torchrun --nproc_per_node=8 "${training_script}" \
        "${BASE_ARGS[@]}" "${GPT2_ARGS[@]}" "$@"
    fi
  done
}

run_case "3_none_zero2" \
  --learning_rate 1e-05 \
  --lr_decay_style none \
  --dtype float32

run_case "4_constant_tp4" \
  --learning_rate 1e-05 \
  --min_lr 1e-06 \
  --lr_decay_style constant \
  --lr_warmup_iters 0 \
  --lr_decay_iters 0 \
  --dtype float32

run_case "5_linear_tp4_sp_distopt" \
  --learning_rate 1e-05 \
  --min_lr 1e-06 \
  --lr_decay_style linear \
  --lr_warmup_iters 2 \
  --lr_warmup_init 0.0 \
  --lr_decay_iters 10 \
  --dtype float32

run_case "6_cosine_pp8" \
  --learning_rate 1e-05 \
  --min_lr 1e-06 \
  --lr_decay_style cosine \
  --lr_warmup_iters 2 \
  --lr_warmup_init 0.0 \
  --lr_decay_iters 10 \
  --dtype float32

run_case "7_inverse_sqrt_pp4_vpp2" \
  --learning_rate 1e-05 \
  --min_lr 1e-06 \
  --lr_decay_style inverse-square-root \
  --lr_warmup_iters 2 \
  --lr_warmup_init 0.0 \
  --lr_decay_iters 10 \
  --dtype float32

run_case "8_cosine_all_parallel_distopt" \
  --learning_rate 1e-05 \
  --min_lr 1e-06 \
  --lr_decay_style cosine \
  --lr_warmup_iters 2 \
  --lr_warmup_init 0.0 \
  --lr_decay_iters 10 \
  --dtype float32

run_case "3_bfloat16_linear" \
  --learning_rate 1e-05 \
  --min_lr 1e-06 \
  --lr_decay_style linear \
  --lr_warmup_iters 2 \
  --lr_warmup_init 0.0 \
  --lr_decay_iters 0 \
  --dtype bfloat16

run_case "4_bfloat16_inverse_sqrt_tp4_distopt" \
  --learning_rate 1e-05 \
  --min_lr 1e-06 \
  --lr_decay_style inverse-square-root \
  --lr_warmup_iters 2 \
  --lr_warmup_init 0.0 \
  --lr_decay_iters 10 \
  --dtype bfloat16

run_case "5_bfloat16_constant_tp4_sp" \
  --learning_rate 1e-05 \
  --min_lr 1e-06 \
  --lr_decay_style constant \
  --lr_warmup_iters 0 \
  --lr_decay_iters 10 \
  --dtype bfloat16

run_case "8_bfloat16_none_all_parallel" \
  --learning_rate 1e-05 \
  --lr_decay_style none \
  --dtype bfloat16

echo "=========================================="
echo "All lr scheduler training jobs finished."
echo "=========================================="
