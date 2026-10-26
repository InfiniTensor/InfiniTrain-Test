#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
ARTIFACT_ROOT="${LLAMA3_MEGATRON_ARTIFACT_ROOT:-${REPO_ROOT}/baseline/megatron/artifacts/llama3}"
CASES="${CASES:-llama3_1,llama3_1_bfloat16,llama3_2,llama3_2_bfloat16,llama3_3,llama3_3_bfloat16,llama3_4,llama3_4_bfloat16,llama3_5,llama3_5_bfloat16,llama3_6,llama3_6_bfloat16,llama3_7,llama3_7_bfloat16,llama3_8,llama3_8_bfloat16}"
LOG_STEP_PERFORMANCE="${LOG_STEP_PERFORMANCE:-1}"
AGGREGATE_LOG="${RESULT_LOG:-${REPO_ROOT}/baseline/megatron/logs/qy_a100_g3025/result_megatron_training_llama3.log}"
mkdir -p "$(dirname "${AGGREGATE_LOG}")" "${ARTIFACT_ROOT}/logs"
: > "${AGGREGATE_LOG}"

case_selected() {
  [[ ",${CASES}," == *",$1,"* ]]
}

run_case() {
  local name="$1"
  local dtype="$2"
  local nproc_per_node="$3"
  local micro_batch_size="$4"
  local total_batch_tokens="$5"
  local tp="${6:-1}"
  local pp="${7:-1}"
  local sp="${8:-0}"
  local vpp="${9:-1}"
  local transformer_impl="${10:-local}"
  local display_dtype="${dtype}"
  if [[ "${dtype}" == "autocast_bfloat16" ]]; then display_dtype="bfloat16"; fi

  if ! case_selected "${name}"; then
    return
  fi

  local case_log="${ARTIFACT_ROOT}/logs/${name}.log"
  : > "${case_log}"
  {
    echo "=============================================="
    echo "Running: ${name} (${display_dtype})"
    echo "=============================================="
  } | tee -a "${case_log}" "${AGGREGATE_LOG}"
  DTYPE="${dtype}" \
  NPROC_PER_NODE="${nproc_per_node}" \
  TP="${tp}" \
  PP="${pp}" \
  SP="${sp}" \
  VPP="${vpp}" \
  TRANSFORMER_IMPL="${transformer_impl}" \
  MICRO_BATCH_SIZE="${micro_batch_size}" \
  TOTAL_BATCH_TOKENS="${total_batch_tokens}" \
  SEQ_LENGTH=64 \
  TRAIN_ITERS=10 \
  LR=1e-5 \
  LOG_STEP_PERFORMANCE="${LOG_STEP_PERFORMANCE}" \
  RESULT_LOG="${case_log}" \
  APPEND_RESULT_LOG=1 \
    bash "${SCRIPT_DIR}/run_training.sh" | tee >(grep --line-buffered "lm loss:" >> "${AGGREGATE_LOG}")
}

# Keep these case names and batch shapes aligned with
# baseline/pytorch/scripts/run_pytorch_training.sh.
run_case llama3_1          float32  1  4  256
run_case llama3_1_bfloat16 autocast_bfloat16 1  4  256
run_case llama3_2          float32  1 80 5120
run_case llama3_2_bfloat16 autocast_bfloat16 1 80 5120
run_case llama3_3          float32  8 10 5120
run_case llama3_3_bfloat16 autocast_bfloat16 8 10 5120
run_case llama3_4          float32  8 40 5120 4 1 0 1
run_case llama3_4_bfloat16 autocast_bfloat16 8 40 5120 4 1 0 1
run_case llama3_5          float32  8 40 5120 4 1 1 1 transformer_engine
run_case llama3_5_bfloat16 autocast_bfloat16 8 40 5120 4 1 1 1 transformer_engine
run_case llama3_6          float32  8 10 5120 1 8 0 1
run_case llama3_6_bfloat16 autocast_bfloat16 8 10 5120 1 8 0 1
run_case llama3_7          float32  4 10 5120 1 4 0 2
run_case llama3_7_bfloat16 autocast_bfloat16 4 10 5120 1 4 0 2
run_case llama3_8          float32  8 40 5120 2 2 0 1
run_case llama3_8_bfloat16 autocast_bfloat16 8 40 5120 2 2 0 1

echo "All selected Llama3 Megatron basic cases finished."
