#!/usr/bin/env bash
set -euo pipefail

# CASES selects A100 case names. NPROC_PER_NODE=1 selects the single-device cases.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
nproc=${NPROC_PER_NODE:-8}
case "$nproc" in
  1|8) ;;
  *) printf 'NPROC_PER_NODE must be 1 or 8.\n' >&2; exit 1 ;;
esac
log_name=result_pytorch_training.log
completion_message="All training jobs finished."
case_suffix='(_bfloat16)?'
lora=()
if [[ ${1:-} == --lora ]]; then
  lora=(--lora_rank 8 --lora_alpha 16.0 --lora_target_modules c_attn,attn.c_proj,c_fc)
  log_name=result_pytorch_training_lora.log
  completion_message="All LoRA training jobs finished."
  case_suffix='_lora_(fp32|bfloat16)'
fi
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HOME="${HF_HOME:-${REPO_ROOT}/data/hf-home}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
GPT2_INPUT_BIN="${GPT2_INPUT_BIN:-${REPO_ROOT}/data/llmc/gpt2/tinyshakespeare/tiny_shakespeare_train.bin}"
LLAMA3_INPUT_BIN="${LLAMA3_INPUT_BIN:-${REPO_ROOT}/data/llmc/llama3/tinyshakespeare/tiny_shakespeare_train.bin}"
GPT2_MODEL="${GPT2_MODEL:-gpt2}"
LLAMA3_MODEL="${LLAMA3_MODEL:-meta-llama/Llama-3.2-1B}"

max_case=3
if [[ $nproc == 1 ]]; then max_case=2; fi
case_pattern="^(gpt2|llama3)_[1-$max_case]${case_suffix}$"
run_gpt2=0
run_llama3=0
IFS=, read -r -a requested_cases <<< "${CASES:-}"
for requested in "${requested_cases[@]}"; do
  if [[ ! $requested =~ $case_pattern ]]; then
    printf 'Unknown or unavailable case: %s\n' "$requested" >&2
    exit 1
  fi
  case "$requested" in
    gpt2_*) run_gpt2=1 ;;
    llama3_*) run_llama3=1 ;;
  esac
done

check_model_inputs() {
  local input=$1 model=$2
  if [[ ! -f $input || ! -r $input ]]; then
    printf 'Input file is missing or unreadable: %s\n' "$input" >&2
    exit 1
  fi
  case "$model" in
    /*|./*|../*)
      if [[ ! -f $model/config.json || ! -r $model/config.json ]]; then
        printf 'Model config is missing or unreadable: %s/config.json\n' "$model" >&2
        exit 1
      fi
      ;;
  esac
}

if [[ -z ${CASES:-} || $run_gpt2 == 1 ]]; then
  check_model_inputs "$GPT2_INPUT_BIN" "$GPT2_MODEL"
fi
if [[ -z ${CASES:-} || $run_llama3 == 1 ]]; then
  check_model_inputs "$LLAMA3_INPUT_BIN" "$LLAMA3_MODEL"
fi

run_model() {
  local model=$1 script=$2 input=$3 model_path=$4
  local index batch total_batch_size dtype name precision details
  local -a launcher
  for index in 1 2 3; do
    launcher=(python -u)
    case "$index" in
      1)
        batch=4
        total_batch_size=256
        ;;
      2)
        batch=80
        total_batch_size=5120
        ;;
      3)
        if [[ $nproc != 8 ]]; then continue; fi
        batch=10
        total_batch_size=5120
        launcher=(torchrun --standalone --nproc_per_node=8)
        ;;
    esac
    for dtype in float32 bfloat16; do
      precision=fp32
      name="${model}_${index}"
      if [[ $dtype == bfloat16 ]]; then
        precision=bfloat16
        name="${name}_bfloat16"
      fi
      if (( ${#lora[@]} )); then name="${model}_${index}_lora_${precision}"; fi
      if [[ -n ${CASES:-} && ",${CASES}," != *",$name,"* ]]; then continue; fi
      details=
      if (( ! ${#lora[@]} )) && [[ $dtype == float32 ]]; then
        details=fp32
      fi
      if [[ $index == 3 ]]; then
        details="${details:+$details, }8 GPUs"
      fi
      echo "=========================================="
      echo "Running: $name${details:+ ($details)}"
      echo "=========================================="
      "${launcher[@]}" "${SCRIPT_DIR}/${script}" \
        --device cuda --dtype "$dtype" --input_bin "$input" --write_tensors 0 \
        --model "$model_path" --batch_size "$batch" --sequence_length 64 --total_batch_size "$total_batch_size" \
        --num_iterations 10 "${lora[@]}"
    done
  done
}

run_cases() {
  run_model gpt2 train_gpt2_musa.py "$GPT2_INPUT_BIN" "$GPT2_MODEL"
  run_model llama3 train_llama3.2_1B_musa.py "$LLAMA3_INPUT_BIN" "$LLAMA3_MODEL"
  echo "=========================================="
  echo "$completion_message"
  echo "=========================================="
}

archive="${REPO_ROOT}/baseline/pytorch/logs/musa_s5000/${log_name}"
result="${RESULT_LOG:-$archive}"
if [[ $nproc == 1 || -n ${CASES:-} ]] &&
   [[ $(realpath -m -- "$result") == "$(realpath -m -- "$archive")" ]]; then
  printf 'Set RESULT_LOG to a separate path for a partial run.\n' >&2
  exit 1
fi
mkdir -p -- "$(dirname -- "$result")"
temporary=$(mktemp -- "$result.XXXXXX")
trap 'rm -f -- "$temporary"' EXIT
run_cases | tee "$temporary"
chmod 0644 "$temporary"
mv -fT -- "$temporary" "$result"
