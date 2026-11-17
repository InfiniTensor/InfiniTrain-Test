#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${TRAINING_DIR}/../.." && pwd)"
cd "${TRAINING_DIR}"

# MLU mirror of ../run_pytorch_training_lr_scheduler.sh (case-for-case).
# Differences vs the CUDA original: mlu/ training scripts, per-script data
# argument injection (the MLU scripts take explicit data/weights paths).

# This script is used to compare loss only.
# The loss performance of each case should be mathematically equivalent to the following setting with 8-card DDP.
BASE_ARGS=(
  --batch_size 10
  --total_batch_size 5120
  --num_iterations 10
)

TRAINING_SCRIPTS=(
  mlu/train_gpt2_mlu.py
  mlu/train_llama3.2_1B_mlu.py
)

# Defaults are resolved relative to the repo root; override any of them via
# the environment when the assets live elsewhere (e.g. a shared NFS mount).
GPT2_INPUT_BIN="${GPT2_INPUT_BIN:-${REPO_ROOT}/data/llmc/gpt2/tinyshakespeare/tiny_shakespeare_train.bin}"
LLAMA3_INPUT_BIN="${LLAMA3_INPUT_BIN:-${REPO_ROOT}/data/llmc/llama3/tinyshakespeare/tiny_shakespeare_train.bin}"
LLAMA_WEIGHTS="${LLAMA_WEIGHTS:-${REPO_ROOT}/data/models/Llama-3.2-1B}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/data/hf-home}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

for path in "${GPT2_INPUT_BIN}" "${LLAMA3_INPUT_BIN}"; do
  [[ -f "${path}" ]] || { echo "Error: input file not found: ${path}" >&2; exit 1; }
done
[[ -d "${LLAMA_WEIGHTS}" ]] || { echo "Error: model directory not found: ${LLAMA_WEIGHTS}" >&2; exit 1; }

data_args_for() {
  case "$1" in
    mlu/train_gpt2_mlu.py)
      echo "--input_bin ${GPT2_INPUT_BIN} --write_tensors 0"
      ;;
    mlu/train_llama3.2_1B_mlu.py)
      echo "--input_bin ${LLAMA3_INPUT_BIN} --model ${LLAMA_WEIGHTS}"
      ;;
  esac
}

run_case() {
  local case_id="$1"
  shift

  for training_script in "${TRAINING_SCRIPTS[@]}"; do
    echo "=========================================="
    echo "Running: ${training_script} / ${case_id}"
    echo "=========================================="
    torchrun --nproc_per_node=8 "${training_script}" $(data_args_for "${training_script}") "${BASE_ARGS[@]}" "$@"
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
