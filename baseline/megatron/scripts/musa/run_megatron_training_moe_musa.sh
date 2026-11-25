#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${BASELINE}/../.." && pwd)"
export PYTHONUNBUFFERED=1
MIXTRAL_WEIGHTS_PATH="${MIXTRAL_WEIGHTS_PATH:-${REPO_ROOT}/data/models/mixtral_megatron_export.bin}"
MIXTRAL_INPUT_BIN="${MIXTRAL_INPUT_BIN:-${LLAMA3_INPUT_BIN:-${REPO_ROOT}/data/llmc/llama3/tinyshakespeare/tiny_shakespeare_train.bin}}"
run_case() {
  local name=$1 dtype=$2 batch=$3
  if [[ -n ${CASES:-} && ",${CASES}," != *",$name,"* ]]; then return; fi
  echo "=============================================="
  echo "Running: $name ($dtype)"
  echo "=============================================="
  python -u "${SCRIPT_DIR}/train_megatron_mixtral_musa.py" \
    --weights_path "$MIXTRAL_WEIGHTS_PATH" --dtype "$dtype" --micro_batch_size "$batch" \
    --global_batch_size "$batch" --num_iterations 10 --log_interval 1 --print_timing \
    --input_bin "$MIXTRAL_INPUT_BIN"
}

valid_cases=mixtral_1,mixtral_1_bfloat16,mixtral_2,mixtral_2_bfloat16
IFS=, read -r -a requested_cases <<< "${CASES:-}"
for requested in "${requested_cases[@]}"; do
  if [[ -z $requested || ",$valid_cases," != *",$requested,"* ]]; then
    printf 'Unknown or unavailable case: %s\n' "$requested" >&2
    exit 1
  fi
done
for path in "$MIXTRAL_INPUT_BIN" "$MIXTRAL_WEIGHTS_PATH"; do
  if [[ ! -f $path || ! -r $path ]]; then
    printf 'Input file is missing or unreadable: %s\n' "$path" >&2
    exit 1
  fi
done

archive="${BASELINE}/logs/musa_s5000/result_megatron_training_moe.log"
result="${RESULT_LOG:-$archive}"
if [[ -n ${CASES:-} ]] &&
   [[ $(realpath -m -- "$result") == "$(realpath -m -- "$archive")" ]]; then
  printf 'Set RESULT_LOG to a separate path for a partial run.\n' >&2
  exit 1
fi
mkdir -p -- "$(dirname -- "$result")"
temporary=$(mktemp -- "$result.XXXXXX")
trap 'rm -f -- "$temporary"' EXIT
{
  run_case mixtral_1 float32 4
  run_case mixtral_1_bfloat16 bfloat16 4
  run_case mixtral_2 float32 80
  run_case mixtral_2_bfloat16 bfloat16 80
} | tee "$temporary"
chmod 0644 "$temporary"
mv -fT -- "$temporary" "$result"
