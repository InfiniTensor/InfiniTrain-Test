#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${BASELINE}/../.." && pwd)"
export PYTHON_EXEC="${SCRIPT_DIR}/megatron_musa.py"
export GPT2_INPUT_BIN="${GPT2_INPUT_BIN:-${REPO_ROOT}/data/llmc/gpt2/tinyshakespeare/tiny_shakespeare_train.bin}"
export GPT2_LLMC_FILEPATH="${GPT2_LLMC_FILEPATH:-${REPO_ROOT}/data/llmc/gpt2/gpt2_124M.bin}"
export PYTHONUNBUFFERED=1
export MUSA_BASELINE_WORKER_ARGS='["--no-persist-layer-norm"]'
export GPT2_MEGATRON_ARTIFACT_ROOT="${GPT2_MEGATRON_ARTIFACT_ROOT:-${BASELINE}/artifacts/gpt2/musa_s5000}"
export GPT2_MEGATRON_DATA_PREFIX="${GPT2_MEGATRON_DATA_PREFIX:-${GPT2_MEGATRON_ARTIFACT_ROOT}/datasets/gpt2}"
export GPT2_MEGATRON_CACHE_PATH="${GPT2_MEGATRON_CACHE_PATH:-${GPT2_MEGATRON_ARTIFACT_ROOT}/cache}"

default_cases=gpt2_1,gpt2_1_bfloat16,gpt2_2,gpt2_2_bfloat16
IFS=, read -r -a requested_cases <<< "${CASES:-}"
for requested in "${requested_cases[@]}"; do
  if [[ -z $requested || ",$default_cases," != *",$requested,"* ]]; then
    printf 'Unknown or unavailable case: %s\n' "$requested" >&2
    exit 1
  fi
done
for path in "$GPT2_INPUT_BIN" "$GPT2_LLMC_FILEPATH"; do
  if [[ ! -f $path || ! -r $path ]]; then
    printf 'Input file is missing or unreadable: %s\n' "$path" >&2
    exit 1
  fi
done

archive="${BASELINE}/logs/musa_s5000/result_megatron_training_gpt2.log"
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
bash "${SCRIPT_DIR}/../gpt2/prepare.sh"
RESULT_LOG="$temporary" bash "${SCRIPT_DIR}/../gpt2/run_cases.sh"
chmod 0644 "$temporary"
mv -fT -- "$temporary" "$result"
