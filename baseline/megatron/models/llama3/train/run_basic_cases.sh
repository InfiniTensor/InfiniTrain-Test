#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
ARTIFACT_ROOT="${LLAMA3_MEGATRON_ARTIFACT_ROOT:-${REPO_ROOT}/baseline/megatron/artifacts/llama3}"
CASES="${CASES:-llama3_1,llama3_1_bfloat16,llama3_2,llama3_2_bfloat16,llama3_3,llama3_3_bfloat16}"
LOG_STEP_PERFORMANCE="${LOG_STEP_PERFORMANCE:-1}"

case_selected() {
  [[ ",${CASES}," == *",$1,"* ]]
}

run_case() {
  local name="$1"
  local dtype="$2"
  local nproc_per_node="$3"
  local micro_batch_size="$4"
  local total_batch_tokens="$5"

  if ! case_selected "${name}"; then
    return
  fi

  echo "=========================================="
  echo "Running: ${name} (${dtype}, ${nproc_per_node} GPU(s))"
  echo "=========================================="
  DTYPE="${dtype}" \
  NPROC_PER_NODE="${nproc_per_node}" \
  TP=1 \
  PP=1 \
  MICRO_BATCH_SIZE="${micro_batch_size}" \
  TOTAL_BATCH_TOKENS="${total_batch_tokens}" \
  SEQ_LENGTH=64 \
  TRAIN_ITERS=10 \
  LR=1e-5 \
  LOG_STEP_PERFORMANCE="${LOG_STEP_PERFORMANCE}" \
  RESULT_LOG="${ARTIFACT_ROOT}/logs/${name}.log" \
    bash "${SCRIPT_DIR}/run_training.sh"
}

# Keep these case names and batch shapes aligned with
# baseline/pytorch/scripts/run_pytorch_training.sh.
run_case llama3_1          float32  1  4  256
run_case llama3_1_bfloat16 autocast_bfloat16 1  4  256
run_case llama3_2          float32  1 80 5120
run_case llama3_2_bfloat16 autocast_bfloat16 1 80 5120
run_case llama3_3          float32  8 10 5120
run_case llama3_3_bfloat16 autocast_bfloat16 8 10 5120

echo "All selected Llama3 Megatron basic cases finished."
