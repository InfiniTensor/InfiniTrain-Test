#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RESULT_LOG="${RESULT_LOG:-${SCRIPT_DIR}/../logs/result_megatron_tiny_mixtral.log}"
INPUT_BIN="${MIXTRAL_INPUT_BIN:-/data1/shared/InfiniTrain-dev/data/llmc/llama3/tinyshakespeare/tiny_shakespeare_train.bin}"
MODEL_DIR="${MEGATRON_MODEL_DIR:-${SCRIPT_DIR}/../models}"
WEIGHTS_PATH="${MODEL_DIR}/tiny_mixtral_megatron_export.bin"
MEGATRON_PATH="${MEGATRON_PATH:-${REPO_ROOT}/third_party/Megatron-LM}"
MASTER_PORT_BASE="${MEGATRON_MASTER_PORT_BASE:-29571}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$(dirname "${RESULT_LOG}")" "${MODEL_DIR}"
: > "${RESULT_LOG}"

echo "=============================================="
echo "Generating: tiny_mixtral Megatron LLMC bin"
echo "=============================================="
python "${SCRIPT_DIR}/train_megatron_tiny_mixtral.py" \
    --write_model_path "${WEIGHTS_PATH}" \
    --megatron_path "${MEGATRON_PATH}" \
    --num_iterations 0 \
    --master_port "${MASTER_PORT_BASE}" \
    | tee -a "${RESULT_LOG}"

run_case() {
    local case_id="$1"
    local dtype="$2"
    local micro_batch_size="$3"
    local global_batch_size="$4"
    local num_iterations="$5"
    local master_port="$6"

    {
        echo ""
        echo "=============================================="
        echo "Running: tiny_mixtral_${case_id} (${dtype})"
        echo "=============================================="
        python "${SCRIPT_DIR}/train_megatron_tiny_mixtral.py" \
            --weights_path "${WEIGHTS_PATH}" \
            --megatron_path "${MEGATRON_PATH}" \
            --input_bin "${INPUT_BIN}" \
            --dtype "${dtype}" \
            --micro_batch_size "${micro_batch_size}" \
            --global_batch_size "${global_batch_size}" \
            --num_iterations "${num_iterations}" \
            --master_port "${master_port}" \
            --log_interval 1 \
            --print_timing
    } | tee -a "${RESULT_LOG}"
}

run_case "1" "float32" 4 4 10 "$((MASTER_PORT_BASE + 0))"
run_case "1_bfloat16" "bfloat16" 4 4 10 "$((MASTER_PORT_BASE + 1))"
run_case "2" "float32" 80 80 10 "$((MASTER_PORT_BASE + 2))"
run_case "2_bfloat16" "bfloat16" 80 80 10 "$((MASTER_PORT_BASE + 3))"

echo "=========================================="
echo "Megatron tiny Mixtral jobs finished. Log: ${RESULT_LOG}"
echo "=========================================="
