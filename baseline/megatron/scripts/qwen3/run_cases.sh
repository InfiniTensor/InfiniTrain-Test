#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
ARTIFACT_ROOT="${QWEN3_MEGATRON_ARTIFACT_ROOT:-${REPO_ROOT}/baseline/megatron/artifacts/qwen3}"
CASES="${CASES:-qwen3_tp8_fp32,qwen3_tp8_bfloat16,qwen3_tp8_sp_fp32,qwen3_pp8_fp32,qwen3_3d_ddp_tp2_pp2_fp32,qwen3_3d_ddp_tp2_pp2_bf16}"
LOG_STEP_PERFORMANCE="${LOG_STEP_PERFORMANCE:-1}"
AGGREGATE_LOG="${RESULT_LOG:-${REPO_ROOT}/baseline/megatron/logs/qy_a100_g3025/result_megatron_training_qwen3.log}"
mkdir -p "$(dirname "${AGGREGATE_LOG}")" "${ARTIFACT_ROOT}/logs"
: > "${AGGREGATE_LOG}"

case_selected() {
  [[ ",${CASES}," == *",$1,"* ]]
}

filter_summary_log() {
  awk '
    /^InfiniTrain-Test commit:/ ||
    /^Megatron-LM commit:/ ||
    /^Framework:/ ||
    /^Model:/ ||
    /^Config:/ ||
    /^step[[:space:]]+[0-9]+\/[0-9]+[[:space:]]+\|[[:space:]]+train loss/ ||
    /^final [0-9]+ iters (Latency|Throughput) avg:/ ||
    /^peak memory consumption:/ {
      print
    }
  '
}

run_case() {
  local name="$1"
  local dtype="$2"
  local micro_batch_size="$3"
  local total_batch_tokens="$4"
  local tp="${5:-1}"
  local pp="${6:-1}"
  local sp="${7:-0}"
  local vpp="${8:-1}"
  local pipeline_layout="${9:-}"
  local transformer_impl="${10:-local}"
  local hybrid_local="${11:-0}"

  if ! case_selected "${name}"; then
    return
  fi

  local display_dtype="${dtype}"
  if [[ "${dtype}" == "autocast_bfloat16" ]]; then display_dtype="bfloat16"; fi
  local case_log="${ARTIFACT_ROOT}/logs/${name}.log"
  : > "${case_log}"
  {
    echo "=============================================="
    echo "Running: ${name} (${display_dtype})"
    echo "=============================================="
  } | tee -a "${case_log}" "${AGGREGATE_LOG}"
  DTYPE="${dtype}" \
  NPROC_PER_NODE=8 \
  TP="${tp}" \
  PP="${pp}" \
  SP="${sp}" \
  VPP="${vpp}" \
  PIPELINE_MODEL_PARALLEL_LAYOUT="${pipeline_layout}" \
  TRANSFORMER_IMPL="${transformer_impl}" \
  QWEN3_HYBRID_LOCAL="${hybrid_local}" \
  MICRO_BATCH_SIZE="${micro_batch_size}" \
  TOTAL_BATCH_TOKENS="${total_batch_tokens}" \
  SEQ_LENGTH=64 \
  TRAIN_ITERS=10 \
  LR=1e-5 \
  LOG_STEP_PERFORMANCE="${LOG_STEP_PERFORMANCE}" \
  RESULT_LOG="${case_log}" \
  APPEND_RESULT_LOG=1 \
    bash "${SCRIPT_DIR}/run.sh" | filter_summary_log | tee -a "${AGGREGATE_LOG}"
}

# Keep case names, iteration counts, batch shapes, and parallel topology aligned
# with the non-LoRA cases in the InfiniTrain qwen3 test group.
# The LoRA-only case is excluded because this baseline loads full weights.
run_case qwen3_tp8_fp32                 float32           1   64 8 1 0 1
run_case qwen3_tp8_bfloat16             autocast_bfloat16 1   64 8 1 0 1
run_case qwen3_tp8_sp_fp32              float32           1   64 8 1 1 1 "" local 1
run_case qwen3_pp8_fp32                 float32           1   64 1 8 0 1 "Et*5|t*5|t*5|t*5|t*4|t*4|t*4|t*4L"
run_case qwen3_3d_ddp_tp2_pp2_fp32      float32          20 5120 2 2 1 2 "" local 1
run_case qwen3_3d_ddp_tp2_pp2_bf16      autocast_bfloat16 20 5120 2 2 1 2 "" local 1

echo "All selected Qwen3 Megatron basic cases finished."
