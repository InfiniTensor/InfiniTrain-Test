#!/usr/bin/env bash
set -euo pipefail

### Set HF-related environments correctly if needed
# export HF_HOME=/data1/shared/InfiniTrain-dev/env/HuggingFace
# export HF_HUB_CACHE=/data1/shared/InfiniTrain-dev/env/HuggingFace/hub
# export HF_HUB_OFFLINE=1

# ---------- GPT-2 ----------
echo "=========================================="
echo "Running: gpt2_1 (fp32)"
echo "=========================================="
python train_gpt2.py --dtype float32

echo "=========================================="
echo "Running: gpt2_1_bfloat16"
echo "=========================================="
python train_gpt2.py --dtype bfloat16

echo "=========================================="
echo "Running: gpt2_2 (fp32)"
echo "=========================================="
python train_gpt2.py \
  --batch_size 80 \
  --total_batch_size 5120 \
  --num_iterations 10 \
  --dtype float32

echo "=========================================="
echo "Running: gpt2_2_bfloat16"
echo "=========================================="
python train_gpt2.py \
  --batch_size 80 \
  --total_batch_size 5120 \
  --num_iterations 10 \
  --dtype bfloat16

echo "=========================================="
echo "Running: gpt2_3 (fp32, 8 GPUs)"
echo "=========================================="
torchrun --nproc_per_node=8 train_gpt2.py \
  --batch_size 10 \
  --total_batch_size 5120 \
  --num_iterations 10 \
  --dtype float32

echo "=========================================="
echo "Running: gpt2_3_bfloat16 (8 GPUs)"
echo "=========================================="
torchrun --nproc_per_node=8 train_gpt2.py \
  --batch_size 10 \
  --total_batch_size 5120 \
  --num_iterations 10 \
  --dtype bfloat16

# ---------- LLaMA 3.2 1B ----------
echo "=========================================="
echo "Running: llama3_1 (fp32)"
echo "=========================================="
python train_llama3.2_1B.py --dtype float32

echo "=========================================="
echo "Running: llama3_1_bfloat16"
echo "=========================================="
python train_llama3.2_1B.py --dtype bfloat16

echo "=========================================="
echo "Running: llama3_2 (fp32)"
echo "=========================================="
python train_llama3.2_1B.py \
  --batch_size 80 \
  --total_batch_size 5120 \
  --num_iterations 10 \
  --dtype float32

echo "=========================================="
echo "Running: llama3_2_bfloat16"
echo "=========================================="
python train_llama3.2_1B.py \
  --batch_size 80 \
  --total_batch_size 5120 \
  --num_iterations 10 \
  --dtype bfloat16

echo "=========================================="
echo "Running: llama3_3 (fp32, 8 GPUs)"
echo "=========================================="
torchrun --nproc_per_node=8 train_llama3.2_1B.py \
  --batch_size 10 \
  --total_batch_size 5120 \
  --num_iterations 10 \
  --dtype float32

echo "=========================================="
echo "Running: llama3_3_bfloat16 (8 GPUs)"
echo "=========================================="
torchrun --nproc_per_node=8 train_llama3.2_1B.py \
  --batch_size 10 \
  --total_batch_size 5120 \
  --num_iterations 10 \
  --dtype bfloat16

# ---------- FM9G4B-V ----------
echo "=========================================="
echo "Running: fm9g4bv_1_bfloat16 (single GPU, frozen LLM)"
echo "=========================================="
# Restrict Trainer to one GPU, respecting an existing visible-device list.
fm9g_single_device="${CUDA_VISIBLE_DEVICES:-0}"
CUDA_VISIBLE_DEVICES="${fm9g_single_device%%,*}" python train_fm9g4bv.py --tune_llm false


echo "=========================================="
echo "Running: fm9g4bv_1_fp32 (single GPU, frozen LLM)"
echo "=========================================="
CUDA_VISIBLE_DEVICES=0 python train_fm9g4bv.py \
  --bf16 false \
  --fp16 false \
  --tune_llm false

echo "=========================================="
echo "Running: fm9g4bv_2_bfloat16 (8 GPUs, full training, ZeRO-2)"
echo "=========================================="
torchrun --nnodes=1 --nproc_per_node=8 \
  --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
  train_fm9g4bv.py --zero2 true

echo "=========================================="
echo "Running: fm9g4bv_2_fp32 (8 GPUs, full training, ZeRO-2)"
echo "=========================================="
torchrun --nnodes=1 --nproc_per_node=8 \
  --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
  train_fm9g4bv.py \
  --zero2 true \
  --bf16 false \
  --fp16 false

echo "=========================================="
echo "All training jobs finished."
echo "=========================================="
