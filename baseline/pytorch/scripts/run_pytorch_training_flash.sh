#!/usr/bin/env bash
set -euo pipefail

### Set HF-related environments correctly if needed
# export HF_HOME=/data1/shared/InfiniTrain-dev/env/HuggingFace
# export HF_HUB_CACHE=/data1/shared/InfiniTrain-dev/env/HuggingFace/hub
# export HF_HUB_OFFLINE=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

GPT2_INPUT_BIN="${GPT2_INPUT_BIN:-/data1/shared/InfiniTrain-dev/data/llmc/gpt2/tinyshakespeare/tiny_shakespeare_train.bin}"
LLAMA3_INPUT_BIN="${LLAMA3_INPUT_BIN:-/data1/shared/InfiniTrain-dev/data/llmc/llama3/tinyshakespeare/tiny_shakespeare_train.bin}"

TRAINING_SCRIPTS=(
  train_gpt2.py
  train_llama3.2_1B.py
)

INPUT_BINS=(
  "${GPT2_INPUT_BIN}"
  "${LLAMA3_INPUT_BIN}"
)

COMMON_ARGS=(
  --attention_backend flash
  --dtype bfloat16
  --num_iterations 10
  --write_tensors 0
)

run_case() {
  local case_id="$1"
  local batch_size="$2"
  local sequence_length="$3"
  local total_batch_size="$4"

  for idx in "${!TRAINING_SCRIPTS[@]}"; do
    local training_script="${TRAINING_SCRIPTS[$idx]}"
    local input_bin="${INPUT_BINS[$idx]}"

    echo "=========================================="
    echo "Running: ${training_script} / ${case_id}"
    echo "=========================================="
    torchrun --standalone --nproc_per_node=8 "${training_script}" \
      --input_bin "${input_bin}" \
      --batch_size "${batch_size}" \
      --sequence_length "${sequence_length}" \
      --total_batch_size "${total_batch_size}" \
      "${COMMON_ARGS[@]}"
  done
}

run_case "dp8_bs8_seq256" 8 256 16384
run_case "dp8_bs16_seq256" 16 256 32768
run_case "dp8_bs8_seq512" 8 512 32768
run_case "dp8_bs8_seq1024" 8 1024 65536

echo "=========================================="
echo "All FlashAttention training jobs finished."
echo "=========================================="
