#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${BASELINE}/../.." && pwd)"
export PYTHON_EXEC="${SCRIPT_DIR}/megatron_musa.py"
export LLAMA3_INPUT_BIN="${LLAMA3_INPUT_BIN:-${REPO_ROOT}/data/llmc/llama3/tinyshakespeare/tiny_shakespeare_train.bin}"
export LLAMA3_LLMC_FILEPATH="${LLAMA3_LLMC_FILEPATH:-${REPO_ROOT}/data/llmc/llama3/llama3.2_1B_fp32.bin}"
export PYTHONUNBUFFERED=1
export LLAMA3_MEGATRON_ARTIFACT_ROOT="${LLAMA3_MEGATRON_ARTIFACT_ROOT:-${BASELINE}/artifacts/llama3/musa_s5000}"
export LLAMA3_MEGATRON_DATA_PREFIX="${LLAMA3_MEGATRON_DATA_PREFIX:-${LLAMA3_MEGATRON_ARTIFACT_ROOT}/datasets/llama3}"
export LLAMA3_MEGATRON_CACHE_PATH="${LLAMA3_MEGATRON_CACHE_PATH:-${LLAMA3_MEGATRON_ARTIFACT_ROOT}/cache}"

default_cases=llama3_1,llama3_1_bfloat16,llama3_2,llama3_2_bfloat16
IFS=, read -r -a requested_cases <<< "${CASES:-}"
for requested in "${requested_cases[@]}"; do
  if [[ -z $requested || ",$default_cases," != *",$requested,"* ]]; then
    printf 'Unknown or unavailable case: %s\n' "$requested" >&2
    exit 1
  fi
done
for path in "$LLAMA3_INPUT_BIN" "$LLAMA3_LLMC_FILEPATH"; do
  if [[ ! -f $path || ! -r $path ]]; then
    printf 'Input file is missing or unreadable: %s\n' "$path" >&2
    exit 1
  fi
done

archive="${BASELINE}/logs/musa_s5000/result_megatron_training_llama3.log"
result="${RESULT_LOG:-$archive}"
if [[ -n ${CASES:-} ]] &&
   [[ $(realpath -m -- "$result") == "$(realpath -m -- "$archive")" ]]; then
  printf 'Set RESULT_LOG to a separate path for a partial run.\n' >&2
  exit 1
fi
mkdir -p -- "$(dirname -- "$result")"
temporary=$(mktemp -- "$result.XXXXXX")
trap 'rm -f -- "$temporary"' EXIT
export CASES="${CASES:-$default_cases}"
bash "${SCRIPT_DIR}/../llama3/prepare.sh"
RESULT_LOG="$temporary" bash "${SCRIPT_DIR}/../llama3/run_cases.sh"
chmod 0644 "$temporary"
mv -fT -- "$temporary" "$result"
