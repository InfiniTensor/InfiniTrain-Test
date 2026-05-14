#!/usr/bin/env bash
set -euo pipefail
source /opt/miniconda3/bin/activate pytorch_env
export HF_HOME=/data/shared/InfiniTrain-dev/env/HuggingFace

# Optional: set model cache directory
export HF_HUB_CACHE=/data/shared/InfiniTrain-dev/env/HuggingFace/hub
# Optional: force using local files
export HF_HUB_OFFLINE=1

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
echo "Running: gpt2_1_lora_fp32"
echo "=========================================="
python train_gpt2.py --dtype float32 --lora_rank 8 --lora_alpha 16.0

echo "=========================================="
echo "Running: gpt2_1_lora_bfloat16"
echo "=========================================="
python train_gpt2.py --dtype bfloat16 --lora_rank 8 --lora_alpha 16.0

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
echo "Running: gpt2_2_lora_fp32"
echo "=========================================="
python train_gpt2.py \
  --batch_size 80 \
  --total_batch_size 5120 \
  --num_iterations 10 \
  --dtype float32 \
  --lora_rank 8 --lora_alpha 16.0

echo "=========================================="
echo "Running: gpt2_2_lora_bfloat16"
echo "=========================================="
python train_gpt2.py \
  --batch_size 80 \
  --total_batch_size 5120 \
  --num_iterations 10 \
  --dtype bfloat16 \
  --lora_rank 8 --lora_alpha 16.0

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

echo "=========================================="
echo "Running: gpt2_3_lora_fp32 (8 GPUs)"
echo "=========================================="
torchrun --nproc_per_node=8 train_gpt2.py \
  --batch_size 10 \
  --total_batch_size 5120 \
  --num_iterations 10 \
  --dtype float32 \
  --lora_rank 8 --lora_alpha 16.0

echo "=========================================="
echo "Running: gpt2_3_lora_bfloat16 (8 GPUs)"
echo "=========================================="
torchrun --nproc_per_node=8 train_gpt2.py \
  --batch_size 10 \
  --total_batch_size 5120 \
  --num_iterations 10 \
  --dtype bfloat16 \
  --lora_rank 8 --lora_alpha 16.0

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
echo "Running: llama3_1_lora_fp32"
echo "=========================================="
python train_llama3.2_1B.py --dtype float32 --lora_rank 8 --lora_alpha 16.0

echo "=========================================="
echo "Running: llama3_1_lora_bfloat16"
echo "=========================================="
python train_llama3.2_1B.py --dtype bfloat16 --lora_rank 8 --lora_alpha 16.0

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
echo "Running: llama3_2_lora_fp32"
echo "=========================================="
python train_llama3.2_1B.py \
  --batch_size 80 \
  --total_batch_size 5120 \
  --num_iterations 10 \
  --dtype float32 \
  --lora_rank 8 --lora_alpha 16.0

echo "=========================================="
echo "Running: llama3_2_lora_bfloat16"
echo "=========================================="
python train_llama3.2_1B.py \
  --batch_size 80 \
  --total_batch_size 5120 \
  --num_iterations 10 \
  --dtype bfloat16 \
  --lora_rank 8 --lora_alpha 16.0

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

echo "=========================================="
echo "Running: llama3_3_lora_fp32 (8 GPUs)"
echo "=========================================="
torchrun --nproc_per_node=8 train_llama3.2_1B.py \
  --batch_size 10 \
  --total_batch_size 5120 \
  --num_iterations 10 \
  --dtype float32 \
  --lora_rank 8 --lora_alpha 16.0

echo "=========================================="
echo "Running: llama3_3_lora_bfloat16 (8 GPUs)"
echo "=========================================="
torchrun --nproc_per_node=8 train_llama3.2_1B.py \
  --batch_size 10 \
  --total_batch_size 5120 \
  --num_iterations 10 \
  --dtype bfloat16 \
  --lora_rank 8 --lora_alpha 16.0

echo "=========================================="
echo "All training jobs finished."
echo "=========================================="
