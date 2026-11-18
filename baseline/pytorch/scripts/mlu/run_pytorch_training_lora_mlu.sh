#!/usr/bin/env bash
set -euo pipefail

# MLU mirror of ../run_pytorch_training_lora.sh (case-for-case).
# Differences vs the CUDA original: mlu/ training scripts, explicit
# --input_bin/--model paths with env-overridable defaults, --write_tensors 0
# on GPT-2.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${TRAINING_DIR}/../.." && pwd)"
cd "${TRAINING_DIR}"

# Defaults are resolved relative to the repo root; override any of them via
# the environment when the assets live elsewhere (e.g. a shared NFS mount).
GPT2_INPUT_BIN="${GPT2_INPUT_BIN:-${REPO_ROOT}/data/llmc/gpt2/tinyshakespeare/tiny_shakespeare_train.bin}"
LLAMA3_INPUT_BIN="${LLAMA3_INPUT_BIN:-${REPO_ROOT}/data/llmc/llama3/tinyshakespeare/tiny_shakespeare_train.bin}"
LLAMA_WEIGHTS="${LLAMA_WEIGHTS:-${REPO_ROOT}/data/models/Llama-3.2-1B}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/data/hf-home}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

for path in "${GPT2_INPUT_BIN}" "${LLAMA3_INPUT_BIN}"; do
  [[ -f "${path}" ]] || { echo "Error: input file not found: ${path}" >&2; exit 1; }
done
[[ -d "${LLAMA_WEIGHTS}" ]] || { echo "Error: model directory not found: ${LLAMA_WEIGHTS}" >&2; exit 1; }

LORA_ARGS=(
  --lora_rank 8
  --lora_alpha 16.0
)

# ---------- GPT-2 LoRA ----------
echo "=========================================="
echo "Running: gpt2_1_lora_fp32"
echo "=========================================="
python mlu/train_gpt2_mlu.py --input_bin "${GPT2_INPUT_BIN}" --write_tensors 0 --dtype float32 "${LORA_ARGS[@]}"

echo "=========================================="
echo "Running: gpt2_1_lora_bfloat16"
echo "=========================================="
python mlu/train_gpt2_mlu.py --input_bin "${GPT2_INPUT_BIN}" --write_tensors 0 --dtype bfloat16 "${LORA_ARGS[@]}"

echo "=========================================="
echo "Running: gpt2_2_lora_fp32"
echo "=========================================="
python mlu/train_gpt2_mlu.py --input_bin "${GPT2_INPUT_BIN}" --write_tensors 0 \
  --batch_size 80 \
  --total_batch_size 5120 \
  --num_iterations 10 \
  --dtype float32 \
  "${LORA_ARGS[@]}"

echo "=========================================="
echo "Running: gpt2_2_lora_bfloat16"
echo "=========================================="
python mlu/train_gpt2_mlu.py --input_bin "${GPT2_INPUT_BIN}" --write_tensors 0 \
  --batch_size 80 \
  --total_batch_size 5120 \
  --num_iterations 10 \
  --dtype bfloat16 \
  "${LORA_ARGS[@]}"

echo "=========================================="
echo "Running: gpt2_3_lora_fp32 (8 GPUs)"
echo "=========================================="
torchrun --nproc_per_node=8 mlu/train_gpt2_mlu.py --input_bin "${GPT2_INPUT_BIN}" --write_tensors 0 \
  --batch_size 10 \
  --total_batch_size 5120 \
  --num_iterations 10 \
  --dtype float32 \
  "${LORA_ARGS[@]}"

echo "=========================================="
echo "Running: gpt2_3_lora_bfloat16 (8 GPUs)"
echo "=========================================="
torchrun --nproc_per_node=8 mlu/train_gpt2_mlu.py --input_bin "${GPT2_INPUT_BIN}" --write_tensors 0 \
  --batch_size 10 \
  --total_batch_size 5120 \
  --num_iterations 10 \
  --dtype bfloat16 \
  "${LORA_ARGS[@]}"

# ---------- LLaMA 3.2 1B LoRA ----------
echo "=========================================="
echo "Running: llama3_1_lora_fp32"
echo "=========================================="
python mlu/train_llama3.2_1B_mlu.py --input_bin "${LLAMA3_INPUT_BIN}" --model "${LLAMA_WEIGHTS}" --dtype float32 "${LORA_ARGS[@]}"

echo "=========================================="
echo "Running: llama3_1_lora_bfloat16"
echo "=========================================="
python mlu/train_llama3.2_1B_mlu.py --input_bin "${LLAMA3_INPUT_BIN}" --model "${LLAMA_WEIGHTS}" --dtype bfloat16 "${LORA_ARGS[@]}"

echo "=========================================="
echo "Running: llama3_2_lora_fp32"
echo "=========================================="
python mlu/train_llama3.2_1B_mlu.py --input_bin "${LLAMA3_INPUT_BIN}" --model "${LLAMA_WEIGHTS}" \
  --batch_size 80 \
  --total_batch_size 5120 \
  --num_iterations 10 \
  --dtype float32 \
  "${LORA_ARGS[@]}"

echo "=========================================="
echo "Running: llama3_2_lora_bfloat16"
echo "=========================================="
python mlu/train_llama3.2_1B_mlu.py --input_bin "${LLAMA3_INPUT_BIN}" --model "${LLAMA_WEIGHTS}" \
  --batch_size 80 \
  --total_batch_size 5120 \
  --num_iterations 10 \
  --dtype bfloat16 \
  "${LORA_ARGS[@]}"

echo "=========================================="
echo "Running: llama3_3_lora_fp32 (8 GPUs)"
echo "=========================================="
torchrun --nproc_per_node=8 mlu/train_llama3.2_1B_mlu.py --input_bin "${LLAMA3_INPUT_BIN}" --model "${LLAMA_WEIGHTS}" \
  --batch_size 10 \
  --total_batch_size 5120 \
  --num_iterations 10 \
  --dtype float32 \
  "${LORA_ARGS[@]}"

echo "=========================================="
echo "Running: llama3_3_lora_bfloat16 (8 GPUs)"
echo "=========================================="
torchrun --nproc_per_node=8 mlu/train_llama3.2_1B_mlu.py --input_bin "${LLAMA3_INPUT_BIN}" --model "${LLAMA_WEIGHTS}" \
  --batch_size 10 \
  --total_batch_size 5120 \
  --num_iterations 10 \
  --dtype bfloat16 \
  "${LORA_ARGS[@]}"

echo "=========================================="
echo "All LoRA training jobs finished."
echo "=========================================="
