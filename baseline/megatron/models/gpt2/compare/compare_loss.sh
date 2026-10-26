#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
MEGATRON_BASELINE="${REPO_ROOT}/baseline/megatron"
INFINITRAIN_LOG="${INFINITRAIN_LOG:?set INFINITRAIN_LOG to the InfiniTrain log}"
MEGATRON_LOG="${MEGATRON_LOG:-${MEGATRON_BASELINE}/artifacts/gpt2/logs/result_megatron_training.log}"
OUTPUT_JSON="${OUTPUT_JSON:-${MEGATRON_BASELINE}/artifacts/gpt2/results/loss_comparison.json}"

python "${MEGATRON_BASELINE}/common/tools/compare_loss.py" \
  --infinitrain-log "${INFINITRAIN_LOG}" \
  --megatron-log "${MEGATRON_LOG}" \
  --expected-steps "${EXPECTED_STEPS:-10}" \
  --atol "${ATOL:-2e-3}" \
  --megatron-step2-running-average \
  --output-json "${OUTPUT_JSON}"
