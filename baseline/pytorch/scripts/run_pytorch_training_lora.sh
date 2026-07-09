#!/usr/bin/env bash
set -euo pipefail

### Set HF-related environments correctly if needed
# export HF_HOME=/data1/shared/InfiniTrain-dev/env/HuggingFace
# export HF_HUB_CACHE=/data1/shared/InfiniTrain-dev/env/HuggingFace/hub
# export HF_HUB_OFFLINE=1

LORA_ARGS=(
  --lora_rank 8
  --lora_alpha 16.0
)

# ---------- GPT-2 LoRA ----------
echo "=========================================="
echo "Running: gpt2_1_lora_fp32"
echo "=========================================="
python train_gpt2.py --dtype float32 "${LORA_ARGS[@]}"

echo "=========================================="
echo "Running: gpt2_1_lora_bfloat16"
echo "=========================================="
python train_gpt2.py --dtype bfloat16 "${LORA_ARGS[@]}"

echo "=========================================="
echo "Running: gpt2_2_lora_fp32"
echo "=========================================="
python train_gpt2.py \
  --batch_size 80 \
  --total_batch_size 5120 \
  --num_iterations 10 \
  --dtype float32 \
  "${LORA_ARGS[@]}"

echo "=========================================="
echo "Running: gpt2_2_lora_bfloat16"
echo "=========================================="
python train_gpt2.py \
  --batch_size 80 \
  --total_batch_size 5120 \
  --num_iterations 10 \
  --dtype bfloat16 \
  "${LORA_ARGS[@]}"

echo "=========================================="
echo "Running: gpt2_3_lora_fp32 (8 GPUs)"
echo "=========================================="
torchrun --nproc_per_node=8 train_gpt2.py \
  --batch_size 10 \
  --total_batch_size 5120 \
  --num_iterations 10 \
  --dtype float32 \
  "${LORA_ARGS[@]}"

echo "=========================================="
echo "Running: gpt2_3_lora_bfloat16 (8 GPUs)"
echo "=========================================="
torchrun --nproc_per_node=8 train_gpt2.py \
  --batch_size 10 \
  --total_batch_size 5120 \
  --num_iterations 10 \
  --dtype bfloat16 \
  "${LORA_ARGS[@]}"

# ---------- LLaMA 3.2 1B LoRA ----------
echo "=========================================="
echo "Running: llama3_1_lora_fp32"
echo "=========================================="
python train_llama3.2_1B.py --dtype float32 "${LORA_ARGS[@]}"

echo "=========================================="
echo "Running: llama3_1_lora_bfloat16"
echo "=========================================="
python train_llama3.2_1B.py --dtype bfloat16 "${LORA_ARGS[@]}"

echo "=========================================="
echo "Running: llama3_2_lora_fp32"
echo "=========================================="
python train_llama3.2_1B.py \
  --batch_size 80 \
  --total_batch_size 5120 \
  --num_iterations 10 \
  --dtype float32 \
  "${LORA_ARGS[@]}"

echo "=========================================="
echo "Running: llama3_2_lora_bfloat16"
echo "=========================================="
python train_llama3.2_1B.py \
  --batch_size 80 \
  --total_batch_size 5120 \
  --num_iterations 10 \
  --dtype bfloat16 \
  "${LORA_ARGS[@]}"

echo "=========================================="
echo "Running: llama3_3_lora_fp32 (8 GPUs)"
echo "=========================================="
torchrun --nproc_per_node=8 train_llama3.2_1B.py \
  --batch_size 10 \
  --total_batch_size 5120 \
  --num_iterations 10 \
  --dtype float32 \
  "${LORA_ARGS[@]}"

echo "=========================================="
echo "Running: llama3_3_lora_bfloat16 (8 GPUs)"
echo "=========================================="
torchrun --nproc_per_node=8 train_llama3.2_1B.py \
  --batch_size 10 \
  --total_batch_size 5120 \
  --num_iterations 10 \
  --dtype bfloat16 \
  "${LORA_ARGS[@]}"

echo "=========================================="
echo "All LoRA training jobs finished."
echo "=========================================="
