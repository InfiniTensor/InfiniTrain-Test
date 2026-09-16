#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
ARTIFACT_ROOT="${QWEN3_MEGATRON_ARTIFACT_ROOT:-${REPO_ROOT}/baseline/megatron/artifacts/qwen3}"
CASES="${CASES:-qwen3_tp8_fp32,qwen3_tp8_bfloat16,qwen3_tp8_sp_fp32,qwen3_pp8_fp32}"
TRAIN_ITERS="${TRAIN_ITERS:-2}"
LOG_STEP_PERFORMANCE="${LOG_STEP_PERFORMANCE:-1}"

case_selected() {
  [[ ",${CASES}," == *",$1,"* ]]
}

run_case() {
  local name="$1"
  local dtype="$2"
  local tp="$3"
  local pp="$4"
  local sp="$5"
  local transformer_impl="$6"
  local pipeline_layout="${7:-}"

  if ! case_selected "${name}"; then
    return
  fi

  echo "=========================================="
  echo "Running: ${name} (${dtype}, TP=${tp}, PP=${pp}, SP=${sp})"
  echo "=========================================="
  DTYPE="${dtype}" \
  NPROC_PER_NODE=8 \
  TP="${tp}" \
  PP="${pp}" \
  SP="${sp}" \
  VPP=1 \
  TRANSFORMER_IMPL="${transformer_impl}" \
  PIPELINE_MODEL_PARALLEL_LAYOUT="${pipeline_layout}" \
  MICRO_BATCH_SIZE=1 \
  TOTAL_BATCH_TOKENS=64 \
  SEQ_LENGTH=64 \
  TRAIN_ITERS="${TRAIN_ITERS}" \
  LR=1e-5 \
  LOG_STEP_PERFORMANCE="${LOG_STEP_PERFORMANCE}" \
  RESULT_LOG="${ARTIFACT_ROOT}/logs/${name}.log" \
    bash "${SCRIPT_DIR}/run_training.sh"
}

# Names intentionally match InfiniTrain scripts/test_config.json.
# BF16 uses autocast so parameters remain FP32, matching InfiniTrain semantics.
run_case qwen3_tp8_fp32      float32            8 1 0 local ""
run_case qwen3_tp8_bfloat16  autocast_bfloat16  8 1 0 local ""
run_case qwen3_tp8_sp_fp32   float32            8 1 1 transformer_engine ""
run_case qwen3_pp8_fp32      float32            1 8 0 local "Et*5|t*5|t*5|t*5|t*4|t*4|t*4|t*4L"

echo "All selected Qwen3 Megatron cases finished."
