#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
MEGATRON_BASELINE="${REPO_ROOT}/baseline/megatron"
MEGATRON_PATH="${MEGATRON_PATH:-${REPO_ROOT}/third_party/Megatron-LM}"
INPUT_BIN="${QWEN3_INPUT_BIN:?set QWEN3_INPUT_BIN to the LLMC token file}"
OUTPUT_PREFIX="${QWEN3_MEGATRON_DATA_PREFIX:-${MEGATRON_BASELINE}/artifacts/qwen3/datasets/qwen3_text_document}"

python "${MEGATRON_BASELINE}/common/data/convert_llmc_dataset.py" \
  --input-bin "${INPUT_BIN}" \
  --output-prefix "${OUTPUT_PREFIX}" \
  --megatron-path "${MEGATRON_PATH}"
