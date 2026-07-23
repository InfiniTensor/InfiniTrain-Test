#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${TRAINING_DIR}"

export HF_HOME="${HF_HOME:-/data/shared/InfiniTrain-dev/env/HuggingFace}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export TORCH_DISTRIBUTED_BACKEND="${TORCH_DISTRIBUTED_BACKEND:-nccl}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

PYTHON="${PYTHON:-python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
GPT2_INPUT_BIN="${GPT2_INPUT_BIN:-/nfs/InfiniTrain-dev/data/llmc/gpt2/tinyshakespeare/tiny_shakespeare_train.bin}"
LLAMA3_INPUT_BIN="${LLAMA3_INPUT_BIN:-/nfs/InfiniTrain-dev/data/llmc/llama3/tinyshakespeare/tiny_shakespeare_train.bin}"

for path in "${GPT2_INPUT_BIN}" "${LLAMA3_INPUT_BIN}"; do
  [[ -f "${path}" ]] || { echo "Error: input file not found: ${path}" >&2; exit 1; }
done
command -v "${PYTHON}" >/dev/null 2>&1 || { echo "Error: Python not found: ${PYTHON}" >&2; exit 1; }

run_case() {
  local name="$1"
  local program="$2"
  local input_bin="$3"
  shift 3
  echo "=========================================="
  echo "Running: ${name}"
  echo "=========================================="
  "${PYTHON}" "${program}" --input_bin "${input_bin}" "$@"
}

run_ddp_case() {
  local name="$1"
  local program="$2"
  local input_bin="$3"
  shift 3
  echo "=========================================="
  echo "Running: ${name}"
  echo "=========================================="
  torchrun --nnodes=1 --node_rank=0 \
    --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    "${program}" --input_bin "${input_bin}" --write_tensors 0 "$@"
}

run_model_cases() {
  local prefix="$1"
  local program="$2"
  local input_bin="$3"
  local common=()
  [[ "${prefix}" == "llama3" ]] && common+=(--write_tensors 0)

  run_case "${prefix}_1 (fp32)" "${program}" "${input_bin}" "${common[@]}" --dtype float32
  run_case "${prefix}_1_bfloat16" "${program}" "${input_bin}" "${common[@]}" --dtype bfloat16

  local large_batch=(--batch_size 80 --total_batch_size 5120 --num_iterations 10)
  run_case "${prefix}_2 (fp32)" "${program}" "${input_bin}" "${common[@]}" "${large_batch[@]}" --dtype float32
  run_case "${prefix}_2_bfloat16" "${program}" "${input_bin}" "${common[@]}" "${large_batch[@]}" --dtype bfloat16

  local ddp=(--batch_size 10 --total_batch_size 5120 --num_iterations 10)
  [[ "${prefix}" == "llama3" ]] && ddp+=(--allreduce_chunk_mb 64)
  run_ddp_case "${prefix}_3 (fp32, 8 GPUs)" "${program}" "${input_bin}" "${ddp[@]}" --dtype float32
  run_ddp_case "${prefix}_3_bfloat16 (8 GPUs)" "${program}" "${input_bin}" "${ddp[@]}" --dtype bfloat16
}

run_model_cases gpt2 train_gpt2.py "${GPT2_INPUT_BIN}"
run_model_cases llama3 maca/train_llama3.2_1B.py "${LLAMA3_INPUT_BIN}"

echo "=========================================="
echo "All Maca training jobs finished."
echo "=========================================="
