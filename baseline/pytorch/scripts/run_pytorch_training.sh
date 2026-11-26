#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
GPT2_INPUT_BIN="${GPT2_INPUT_BIN:-${WORKSPACE_ROOT}/InfiniTrain/data/gpt2/tiny_shakespeare_train.bin}"
LLAMA3_INPUT_BIN="${LLAMA3_INPUT_BIN:-${WORKSPACE_ROOT}/InfiniTrain/data/llama3/tiny_shakespeare_train.bin}"
LLAMA3_LLMC_FILEPATH="${LLAMA3_LLMC_FILEPATH:-${WORKSPACE_ROOT}/InfiniTrain/data/llama3/llama3.2_1B_fp32.bin}"
GPT2_ARGS=(
  --input_bin "${GPT2_INPUT_BIN}"
)
LLAMA_ARGS=(
  --input_bin "${LLAMA3_INPUT_BIN}"
  --llmc_filepath "${LLAMA3_LLMC_FILEPATH}"
)

### Set HF-related environments correctly if needed
# export HF_HOME=/data1/shared/InfiniTrain-dev/env/HuggingFace
# export HF_HUB_CACHE=/data1/shared/InfiniTrain-dev/env/HuggingFace/hub
# export HF_HUB_OFFLINE=1

# ---------- GPT-2 ----------
echo "=========================================="
echo "Running: gpt2_1 (fp32)"
echo "=========================================="
python train_gpt2.py --dtype float32 "${GPT2_ARGS[@]}"

echo "=========================================="
echo "Running: gpt2_1_bfloat16"
echo "=========================================="
python train_gpt2.py --dtype bfloat16 "${GPT2_ARGS[@]}"

echo "=========================================="
echo "Running: gpt2_2 (fp32)"
echo "=========================================="
python train_gpt2.py \
  "${GPT2_ARGS[@]}" \
  --batch_size 80 \
  --total_batch_size 5120 \
  --num_iterations 10 \
  --dtype float32

echo "=========================================="
echo "Running: gpt2_2_bfloat16"
echo "=========================================="
python train_gpt2.py \
  "${GPT2_ARGS[@]}" \
  --batch_size 80 \
  --total_batch_size 5120 \
  --num_iterations 10 \
  --dtype bfloat16

echo "=========================================="
echo "Running: gpt2_3 (fp32, 8 GPUs)"
echo "=========================================="
torchrun --nproc_per_node=8 train_gpt2.py \
  "${GPT2_ARGS[@]}" \
  --batch_size 10 \
  --total_batch_size 5120 \
  --num_iterations 10 \
  --dtype float32

echo "=========================================="
echo "Running: gpt2_3_bfloat16 (8 GPUs)"
echo "=========================================="
torchrun --nproc_per_node=8 train_gpt2.py \
  "${GPT2_ARGS[@]}" \
  --batch_size 10 \
  --total_batch_size 5120 \
  --num_iterations 10 \
  --dtype bfloat16

# ---------- LLaMA 3.2 1B ----------
echo "=========================================="
echo "Running: llama3_1 (fp32)"
echo "=========================================="
python train_llama3.2_1B.py --dtype float32 "${LLAMA_ARGS[@]}"

echo "=========================================="
echo "Running: llama3_1_bfloat16"
echo "=========================================="
python train_llama3.2_1B.py --dtype bfloat16 "${LLAMA_ARGS[@]}"

echo "=========================================="
echo "Running: llama3_2 (fp32)"
echo "=========================================="
python train_llama3.2_1B.py \
  "${LLAMA_ARGS[@]}" \
  --batch_size 80 \
  --total_batch_size 5120 \
  --num_iterations 10 \
  --dtype float32

echo "=========================================="
echo "Running: llama3_2_bfloat16"
echo "=========================================="
python train_llama3.2_1B.py \
  "${LLAMA_ARGS[@]}" \
  --batch_size 80 \
  --total_batch_size 5120 \
  --num_iterations 10 \
  --dtype bfloat16

echo "=========================================="
echo "Running: llama3_3 (fp32, 8 GPUs)"
echo "=========================================="
torchrun --nproc_per_node=8 train_llama3.2_1B.py \
  "${LLAMA_ARGS[@]}" \
  --batch_size 10 \
  --total_batch_size 5120 \
  --num_iterations 10 \
  --dtype float32

echo "=========================================="
echo "Running: llama3_3_bfloat16 (8 GPUs)"
echo "=========================================="
torchrun --nproc_per_node=8 train_llama3.2_1B.py \
  "${LLAMA_ARGS[@]}" \
  --batch_size 10 \
  --total_batch_size 5120 \
  --num_iterations 10 \
  --dtype bfloat16

echo "=========================================="
echo "All training jobs finished."
echo "=========================================="
