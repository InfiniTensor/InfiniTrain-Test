#!/usr/bin/env bash
set -euo pipefail

# Match the A100 scheduler cases on 8-device DDP. Compare loss only.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/data/hf-home}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_HUB_DISABLE_PROGRESS_BARS=1
GPT2_INPUT_BIN="${GPT2_INPUT_BIN:-${REPO_ROOT}/data/llmc/gpt2/tinyshakespeare/tiny_shakespeare_train.bin}"
LLAMA3_INPUT_BIN="${LLAMA3_INPUT_BIN:-${REPO_ROOT}/data/llmc/llama3/tinyshakespeare/tiny_shakespeare_train.bin}"
GPT2_MODEL="${GPT2_MODEL:-gpt2}"
LLAMA3_MODEL="${LLAMA3_MODEL:-meta-llama/Llama-3.2-1B}"

case_names=(
  3_none_zero2 4_constant_tp4 5_linear_tp4_sp_distopt 6_cosine_pp8
  7_inverse_sqrt_pp4_vpp2 8_cosine_all_parallel_distopt
  3_bfloat16_linear 4_bfloat16_inverse_sqrt_tp4_distopt
  5_bfloat16_constant_tp4_sp 8_bfloat16_none_all_parallel
)
valid_cases=$(IFS=,; echo "${case_names[*]}")
run_gpt2=0
run_llama3=0
IFS=, read -r -a requested_cases <<< "${CASES:-}"
for requested in "${requested_cases[@]}"; do
  name=${requested##* / }
  if [[ -z $name || ",$valid_cases," != *",$name,"* ]]; then
    printf 'Unknown case: %s\n' "$requested" >&2
    exit 1
  fi
  case "$requested" in
    "$name") run_gpt2=1; run_llama3=1 ;;
    "train_gpt2.py / $name") run_gpt2=1 ;;
    "train_llama3.2_1B.py / $name") run_llama3=1 ;;
    *) printf 'Unknown case: %s\n' "$requested" >&2; exit 1 ;;
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

run_case() {
  local name=$1
  shift
  run_model train_gpt2.py "$GPT2_INPUT_BIN" "$GPT2_MODEL" "$name" "$@"
  run_model train_llama3.2_1B.py "$LLAMA3_INPUT_BIN" "$LLAMA3_MODEL" "$name" "$@"
}

run_model() {
  local script=$1 input=$2 model=$3 name=$4
  local full_name="$script / $name"
  shift 4
  if [[ -n ${CASES:-} && ",${CASES}," != *",$name,"* && ",${CASES}," != *",$full_name,"* ]]; then return; fi
  echo "=========================================="
  echo "Running: ${script} / ${name}"
  echo "=========================================="
  torchrun --standalone --nproc_per_node=8 "${SCRIPT_DIR}/${script%.py}_musa.py" \
    --device cuda --input_bin "$input" --model "$model" --write_tensors 0 \
    --batch_size 10 --sequence_length 64 --total_batch_size 5120 --num_iterations 10 "$@"
}

run_cases() {
  run_case "3_none_zero2" \
    --learning_rate 1e-05 \
    --lr_decay_style none \
    --dtype float32

  run_case "4_constant_tp4" \
    --learning_rate 1e-05 \
    --min_lr 1e-06 \
    --lr_decay_style constant \
    --lr_warmup_iters 0 \
    --lr_decay_iters 0 \
    --dtype float32

  run_case "5_linear_tp4_sp_distopt" \
    --learning_rate 1e-05 \
    --min_lr 1e-06 \
    --lr_decay_style linear \
    --lr_warmup_iters 2 \
    --lr_warmup_init 0.0 \
    --lr_decay_iters 10 \
    --dtype float32

  run_case "6_cosine_pp8" \
    --learning_rate 1e-05 \
    --min_lr 1e-06 \
    --lr_decay_style cosine \
    --lr_warmup_iters 2 \
    --lr_warmup_init 0.0 \
    --lr_decay_iters 10 \
    --dtype float32

  run_case "7_inverse_sqrt_pp4_vpp2" \
    --learning_rate 1e-05 \
    --min_lr 1e-06 \
    --lr_decay_style inverse-square-root \
    --lr_warmup_iters 2 \
    --lr_warmup_init 0.0 \
    --lr_decay_iters 10 \
    --dtype float32

  run_case "8_cosine_all_parallel_distopt" \
    --learning_rate 1e-05 \
    --min_lr 1e-06 \
    --lr_decay_style cosine \
    --lr_warmup_iters 2 \
    --lr_warmup_init 0.0 \
    --lr_decay_iters 10 \
    --dtype float32

  run_case "3_bfloat16_linear" \
    --learning_rate 1e-05 \
    --min_lr 1e-06 \
    --lr_decay_style linear \
    --lr_warmup_iters 2 \
    --lr_warmup_init 0.0 \
    --lr_decay_iters 0 \
    --dtype bfloat16

  run_case "4_bfloat16_inverse_sqrt_tp4_distopt" \
    --learning_rate 1e-05 \
    --min_lr 1e-06 \
    --lr_decay_style inverse-square-root \
    --lr_warmup_iters 2 \
    --lr_warmup_init 0.0 \
    --lr_decay_iters 10 \
    --dtype bfloat16

  run_case "5_bfloat16_constant_tp4_sp" \
    --learning_rate 1e-05 \
    --min_lr 1e-06 \
    --lr_decay_style constant \
    --lr_warmup_iters 0 \
    --lr_decay_iters 10 \
    --dtype bfloat16

  run_case "8_bfloat16_none_all_parallel" \
    --learning_rate 1e-05 \
    --lr_decay_style none \
    --dtype bfloat16

  echo "=========================================="
  echo "All lr scheduler training jobs finished."
  echo "=========================================="
}

archive="${REPO_ROOT}/baseline/pytorch/logs/musa_s5000/result_pytorch_trainng_lr_scheduler.log"
result="${RESULT_LOG:-$archive}"
if [[ -n ${CASES:-} ]] &&
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
