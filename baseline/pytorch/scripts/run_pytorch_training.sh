#!/usr/bin/env bash
set -euo pipefail

### Set HF-related environments correctly if needed
# export HF_HOME=/data1/shared/InfiniTrain-dev/env/HuggingFace
# export HF_HUB_CACHE=/data1/shared/InfiniTrain-dev/env/HuggingFace/hub
# export HF_HUB_OFFLINE=1

# # ---------- GPT-2 ----------
# echo "=========================================="
# echo "Running: gpt2_1 (fp32)"
# echo "=========================================="
# python train_gpt2.py --dtype float32

# echo "=========================================="
# echo "Running: gpt2_1_bfloat16"
# echo "=========================================="
# python train_gpt2.py --dtype bfloat16

# echo "=========================================="
# echo "Running: gpt2_2 (fp32)"
# echo "=========================================="
# python train_gpt2.py \
#   --batch_size 80 \
#   --total_batch_size 5120 \
#   --num_iterations 10 \
#   --dtype float32

# echo "=========================================="
# echo "Running: gpt2_2_bfloat16"
# echo "=========================================="
# python train_gpt2.py \
#   --batch_size 80 \
#   --total_batch_size 5120 \
#   --num_iterations 10 \
#   --dtype bfloat16

# echo "=========================================="
# echo "Running: gpt2_3 (fp32, 8 GPUs)"
# echo "=========================================="
# torchrun --nproc_per_node=8 train_gpt2.py \
#   --batch_size 10 \
#   --total_batch_size 5120 \
#   --num_iterations 10 \
#   --dtype float32

# echo "=========================================="
# echo "Running: gpt2_3_bfloat16 (8 GPUs)"
# echo "=========================================="
# torchrun --nproc_per_node=8 train_gpt2.py \
#   --batch_size 10 \
#   --total_batch_size 5120 \
#   --num_iterations 10 \
#   --dtype bfloat16

# # ---------- LLaMA 3.2 1B ----------
# echo "=========================================="
# echo "Running: llama3_1 (fp32)"
# echo "=========================================="
# python train_llama3.2_1B.py --dtype float32

# echo "=========================================="
# echo "Running: llama3_1_bfloat16"
# echo "=========================================="
# python train_llama3.2_1B.py --dtype bfloat16

# echo "=========================================="
# echo "Running: llama3_2 (fp32)"
# echo "=========================================="
# python train_llama3.2_1B.py \
#   --batch_size 80 \
#   --total_batch_size 5120 \
#   --num_iterations 10 \
#   --dtype float32

# echo "=========================================="
# echo "Running: llama3_2_bfloat16"
# echo "=========================================="
# python train_llama3.2_1B.py \
#   --batch_size 80 \
#   --total_batch_size 5120 \
#   --num_iterations 10 \
#   --dtype bfloat16

# echo "=========================================="
# echo "Running: llama3_3 (fp32, 8 GPUs)"
# echo "=========================================="
# torchrun --nproc_per_node=8 train_llama3.2_1B.py \
#   --batch_size 10 \
#   --total_batch_size 5120 \
#   --num_iterations 10 \
#   --dtype float32

# echo "=========================================="
# echo "Running: llama3_3_bfloat16 (8 GPUs)"
# echo "=========================================="
# torchrun --nproc_per_node=8 train_llama3.2_1B.py \
#   --batch_size 10 \
#   --total_batch_size 5120 \
#   --num_iterations 10 \
#   --dtype bfloat16
  
# ---------- FM9G4B-V ----------
# Set RUN_FM9G4BV=1 and provide these paths to include the multimodal cases.
if [[ "${RUN_FM9G4BV:-0}" == "1" ]]; then
  : "${FM9G_MODEL_PATH:?Set FM9G_MODEL_PATH to the local model directory}"
  : "${FM9G_DATA_PATH:?Set FM9G_DATA_PATH to the training JSON}"
  : "${FM9G_DEEPSPEED_CONFIG:?Set FM9G_DEEPSPEED_CONFIG to the ZeRO-2 JSON}"
  : "${FM9G_OUTPUT_DIR:?Set FM9G_OUTPUT_DIR to an output directory}"
  [[ -d "${FM9G_MODEL_PATH}" ]] || { echo "Model directory not found: ${FM9G_MODEL_PATH}" >&2; exit 1; }
  [[ -f "${FM9G_DATA_PATH}" ]] || { echo "Training data not found: ${FM9G_DATA_PATH}" >&2; exit 1; }
  [[ -f "${FM9G_DEEPSPEED_CONFIG}" ]] || { echo "DeepSpeed config not found: ${FM9G_DEEPSPEED_CONFIG}" >&2; exit 1; }

  fm9g_args=(
    --model_path "${FM9G_MODEL_PATH}"
    --data_path "${FM9G_DATA_PATH}"
    --do_train --bf16 true --tune_vision true
    --max_steps "${FM9G_STEPS:-10}"
    --per_device_train_batch_size 1 --gradient_accumulation_steps 1
    --learning_rate 1e-5 --weight_decay 0.1 --adam_beta2 0.95
    --lr_scheduler_type cosine --warmup_ratio 0.01
    --gradient_checkpointing true --logging_steps 1
    --save_strategy no --report_to none
  )

  echo "=========================================="
  echo "Running: fm9g4bv_1_bfloat16 (single GPU, frozen LLM)"
  echo "=========================================="
  # Trainer otherwise enables DataParallel when multiple GPUs are visible.
  fm9g_single_device="${FM9G_SINGLE_DEVICE:-${CUDA_VISIBLE_DEVICES:-0}}"
  fm9g_single_device="${fm9g_single_device%%,*}"
  CUDA_VISIBLE_DEVICES="${fm9g_single_device}" python train_fm9g4bv.py "${fm9g_args[@]}" --tune_llm false \
    --output_dir "${FM9G_OUTPUT_DIR}/single"

  echo "=========================================="
  echo "Running: fm9g4bv_2_bfloat16 (multi-GPU, full training, ZeRO-2)"
  echo "=========================================="
  torchrun --nnodes=1 --nproc_per_node="${FM9G_NPROC_PER_NODE:-8}" \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
    train_fm9g4bv.py "${fm9g_args[@]}" --tune_llm true \
    --deepspeed "${FM9G_DEEPSPEED_CONFIG}" --output_dir "${FM9G_OUTPUT_DIR}/multi"
fi

echo "=========================================="
echo "All training jobs finished."
echo "=========================================="
