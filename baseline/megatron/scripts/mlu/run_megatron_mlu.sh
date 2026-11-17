#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${SCRIPT_DIR}"

# Defaults are resolved relative to the repo root; override any of them via
# the environment when the assets live elsewhere (e.g. a shared NFS mount).
INPUT_BIN="${MIXTRAL_INPUT_BIN:-${REPO_ROOT}/data/llmc/llama3/tinyshakespeare/tiny_shakespeare_train.bin}"
MODEL_DIR="${MEGATRON_MODEL_DIR:-${SCRIPT_DIR}/../models}"
EXPORT="${MODEL_DIR}/mixtral_megatron_export.bin"
MEGATRON_PATH="${MEGATRON_PATH:-}"  # empty = use the installed megatron.core package
MASTER_PORT_BASE="${MEGATRON_MASTER_PORT_BASE:-29571}"

mkdir -p "${MODEL_DIR}"
[[ -f "${INPUT_BIN}" ]] || { echo "Error: input file not found: ${INPUT_BIN}" >&2; exit 1; }

echo "=========================================="
echo "Exporting init model (num_iterations=0)"
echo "=========================================="
python train_megatron_mixtral_mlu.py \
  --write_model_path "${EXPORT}" \
  --megatron_path "${MEGATRON_PATH}" \
  --num_iterations 0 \
  --master_port "${MASTER_PORT_BASE}"

run_case() {
  local name="$1" dtype="$2" mbs="$3" gbs="$4" port="$5"
  echo "=========================================="
  echo "Running: mixtral_${name} (${dtype})"
  echo "=========================================="
  python train_megatron_mixtral_mlu.py \
    --weights_path "${EXPORT}" \
    --input_bin "${INPUT_BIN}" \
    --dtype "${dtype}" \
    --micro_batch_size "${mbs}" \
    --global_batch_size "${gbs}" \
    --num_iterations 10 \
    --megatron_path "${MEGATRON_PATH}" \
    --master_port "${port}" \
    --log_interval 1 \
    --print_timing
}

run_case "1" "float32" 4 4 "$((MASTER_PORT_BASE + 1))"
run_case "1_bfloat16" "bfloat16" 4 4 "$((MASTER_PORT_BASE + 2))"
run_case "2" "float32" 80 80 "$((MASTER_PORT_BASE + 3))"
run_case "2_bfloat16" "bfloat16" 80 80 "$((MASTER_PORT_BASE + 4))"

echo "=========================================="
echo "All Megatron MLU jobs finished."
echo "=========================================="
