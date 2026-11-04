#!/usr/bin/env bash
set -euo pipefail

# MLU mirror of run_pytorch_training.sh (case-for-case).
# Differences vs the CUDA original: *_mlu training scripts, explicit
# --input_bin/--model paths for the AFS layout, --write_tensors 0 on GPT-2.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
D=/mnt/afs/user_home/wangyichen/infinitrain-baseline
DATA_GPT2="${D}/data/tinyshakespeare/tiny_shakespeare_train.bin"
DATA_LLAMA="${D}/data/tinyshakespeare-llama3/tiny_shakespeare_train.bin"
LLAMA_WEIGHTS="${D}/models/Llama-3.2-1B"
cd "${SCRIPT_DIR}"

# The cluster has no direct internet; resolve hub models from the local
# cache only (populated once via hf-mirror).
export HF_HOME="${D}/models/hf-home"
export HF_HUB_OFFLINE=1

# ---------- GPT-2 ----------
echo "=========================================="
echo "Running: gpt2_1 (fp32)"
echo "=========================================="
python train_gpt2_mlu.py --input_bin "${DATA_GPT2}" --write_tensors 0 --dtype float32

echo "=========================================="
echo "Running: gpt2_1_bfloat16"
echo "=========================================="
python train_gpt2_mlu.py --input_bin "${DATA_GPT2}" --write_tensors 0 --dtype bfloat16

echo "=========================================="
echo "Running: gpt2_2 (fp32)"
echo "=========================================="
python train_gpt2_mlu.py --input_bin "${DATA_GPT2}" --write_tensors 0 \
  --batch_size 80 \
  --total_batch_size 5120 \
  --num_iterations 10 \
  --dtype float32

echo "=========================================="
echo "Running: gpt2_2_bfloat16"
echo "=========================================="
python train_gpt2_mlu.py --input_bin "${DATA_GPT2}" --write_tensors 0 \
  --batch_size 80 \
  --total_batch_size 5120 \
  --num_iterations 10 \
  --dtype bfloat16

echo "=========================================="
echo "Running: gpt2_3 (fp32, 8 GPUs)"
echo "=========================================="
torchrun --nproc_per_node=8 train_gpt2_mlu.py --input_bin "${DATA_GPT2}" --write_tensors 0 \
  --batch_size 10 \
  --total_batch_size 5120 \
  --num_iterations 10 \
  --dtype float32

echo "=========================================="
echo "Running: gpt2_3_bfloat16 (8 GPUs)"
echo "=========================================="
torchrun --nproc_per_node=8 train_gpt2_mlu.py --input_bin "${DATA_GPT2}" --write_tensors 0 \
  --batch_size 10 \
  --total_batch_size 5120 \
  --num_iterations 10 \
  --dtype bfloat16

# ---------- LLaMA 3.2 1B ----------
echo "=========================================="
echo "Running: llama3_1 (fp32)"
echo "=========================================="
python train_llama3.2_1B_mlu.py --input_bin "${DATA_LLAMA}" --model "${LLAMA_WEIGHTS}" --dtype float32

echo "=========================================="
echo "Running: llama3_1_bfloat16"
echo "=========================================="
python train_llama3.2_1B_mlu.py --input_bin "${DATA_LLAMA}" --model "${LLAMA_WEIGHTS}" --dtype bfloat16

echo "=========================================="
echo "Running: llama3_2 (fp32)"
echo "=========================================="
python train_llama3.2_1B_mlu.py --input_bin "${DATA_LLAMA}" --model "${LLAMA_WEIGHTS}" \
  --batch_size 80 \
  --total_batch_size 5120 \
  --num_iterations 10 \
  --dtype float32

echo "=========================================="
echo "Running: llama3_2_bfloat16"
echo "=========================================="
python train_llama3.2_1B_mlu.py --input_bin "${DATA_LLAMA}" --model "${LLAMA_WEIGHTS}" \
  --batch_size 80 \
  --total_batch_size 5120 \
  --num_iterations 10 \
  --dtype bfloat16

echo "=========================================="
echo "Running: llama3_3 (fp32, 8 GPUs)"
echo "=========================================="
torchrun --nproc_per_node=8 train_llama3.2_1B_mlu.py --input_bin "${DATA_LLAMA}" --model "${LLAMA_WEIGHTS}" \
  --batch_size 10 \
  --total_batch_size 5120 \
  --num_iterations 10 \
  --dtype float32

echo "=========================================="
echo "Running: llama3_3_bfloat16 (8 GPUs)"
echo "=========================================="
torchrun --nproc_per_node=8 train_llama3.2_1B_mlu.py --input_bin "${DATA_LLAMA}" --model "${LLAMA_WEIGHTS}" \
  --batch_size 10 \
  --total_batch_size 5120 \
  --num_iterations 10 \
  --dtype bfloat16

echo "=========================================="
echo "All training jobs finished."
echo "=========================================="
