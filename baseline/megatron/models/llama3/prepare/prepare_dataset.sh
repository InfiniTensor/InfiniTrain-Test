#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
BASELINE="${REPO_ROOT}/baseline/megatron"
MEGATRON_PATH="${MEGATRON_PATH:-${REPO_ROOT}/third_party/Megatron-LM}"
INPUT_BIN="${LLAMA3_INPUT_BIN:-/data1/shared/InfiniTrain-dev/data/llmc/llama3/tinyshakespeare/tiny_shakespeare_train.bin}"
OUTPUT_PREFIX="${LLAMA3_MEGATRON_DATA_PREFIX:-${BASELINE}/artifacts/llama3/datasets/llama3_text_document}"

python "${BASELINE}/common/data/convert_llmc_dataset.py" \
  --input-bin "${INPUT_BIN}" --output-prefix "${OUTPUT_PREFIX}" --megatron-path "${MEGATRON_PATH}"
