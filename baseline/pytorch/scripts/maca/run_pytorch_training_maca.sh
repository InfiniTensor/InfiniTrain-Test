#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTORCH_BASELINE_DIR="$(cd -- "${TRAINING_DIR}/.." && pwd)"

export HF_HOME="${HF_HOME:-/data/shared/InfiniTrain-dev/env/HuggingFace}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-${HF_HUB_OFFLINE}}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export TORCH_DISTRIBUTED_BACKEND="${TORCH_DISTRIBUTED_BACKEND:-nccl}"
export TORCH_DIST_BACKEND="${TORCH_DIST_BACKEND:-${TORCH_DISTRIBUTED_BACKEND}}"
export TORCH_CPP_LOG_LEVEL="${TORCH_CPP_LOG_LEVEL:-ERROR}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

PYTHON="${PYTHON:-/opt/conda/bin/python}"
TORCHRUN="${TORCHRUN:-/opt/conda/bin/torchrun}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
GPT2_INPUT_BIN="${GPT2_INPUT_BIN:-/nfs/InfiniTrain-dev/data/llmc/gpt2/tinyshakespeare/tiny_shakespeare_train.bin}"
LLAMA3_INPUT_BIN="${LLAMA3_INPUT_BIN:-/nfs/InfiniTrain-dev/data/llmc/llama3/tinyshakespeare/tiny_shakespeare_train.bin}"
RUN_GPT2="${RUN_GPT2:-1}"
RUN_LLAMA3="${RUN_LLAMA3:-1}"
WRITE_TENSORS="${WRITE_TENSORS:-}"
PLATFORM_NAME="${PLATFORM_NAME:-maca}"
RESULT_LOG="${RESULT_LOG:-${PYTORCH_BASELINE_DIR}/logs/${PLATFORM_NAME}/result_pytorch_training.log}"

[[ "${RUN_GPT2}" =~ ^[01]$ ]] || { echo "Error: RUN_GPT2 must be 0 or 1" >&2; exit 1; }
[[ "${RUN_LLAMA3}" =~ ^[01]$ ]] || { echo "Error: RUN_LLAMA3 must be 0 or 1" >&2; exit 1; }
[[ -z "${WRITE_TENSORS}" || "${WRITE_TENSORS}" =~ ^[01]$ ]] || { echo "Error: WRITE_TENSORS must be 0 or 1" >&2; exit 1; }
(( RUN_GPT2 || RUN_LLAMA3 )) || { echo "Error: no model selected" >&2; exit 1; }
(( ! RUN_GPT2 )) || [[ -f "${GPT2_INPUT_BIN}" ]] || { echo "Error: input file not found: ${GPT2_INPUT_BIN}" >&2; exit 1; }
(( ! RUN_LLAMA3 )) || [[ -f "${LLAMA3_INPUT_BIN}" ]] || { echo "Error: input file not found: ${LLAMA3_INPUT_BIN}" >&2; exit 1; }
[[ -x "${PYTHON}" ]] || { echo "Error: Python not executable: ${PYTHON}" >&2; exit 1; }
[[ -x "${TORCHRUN}" ]] || { echo "Error: torchrun not executable: ${TORCHRUN}" >&2; exit 1; }

for runtime_root in /opt/hpcc-3.5.3 /opt/hpcc /usr/local/maca /opt/maca; do
  if [[ -d "${runtime_root}" ]]; then
    export HPCC_PATH="${HPCC_PATH:-${runtime_root}}"
    export MACA_PATH="${MACA_PATH:-${runtime_root}}"
    break
  fi
done

mkdir -p "$(dirname -- "${RESULT_LOG}")"
if [[ -d /opt/hpcc/lib ]]; then
  cd /opt/hpcc/lib
else
  cd "${TRAINING_DIR}"
fi

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
  "${TORCHRUN}" --nnodes=1 --node_rank=0 \
    --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    "${program}" --input_bin "${input_bin}" "$@"
}

run_model_cases() {
  local prefix="$1"
  local program="$2"
  local input_bin="$3"
  local common=()
  if [[ -n "${WRITE_TENSORS}" ]]; then
    common+=(--write_tensors "${WRITE_TENSORS}")
  elif [[ "${prefix}" == "llama3" ]]; then
    common+=(--write_tensors 0)
  fi

  run_case "${prefix}_1 (fp32)" "${program}" "${input_bin}" "${common[@]}" --dtype float32
  run_case "${prefix}_1_bfloat16" "${program}" "${input_bin}" "${common[@]}" --dtype bfloat16

  local large_batch=(--batch_size 80 --total_batch_size 5120 --num_iterations 10)
  run_case "${prefix}_2 (fp32)" "${program}" "${input_bin}" "${common[@]}" "${large_batch[@]}" --dtype float32
  run_case "${prefix}_2_bfloat16" "${program}" "${input_bin}" "${common[@]}" "${large_batch[@]}" --dtype bfloat16

  local ddp=(--batch_size 10 --total_batch_size 5120 --num_iterations 10)
  [[ "${prefix}" == "llama3" ]] && ddp+=(--allreduce_chunk_mb 64)
  [[ -n "${WRITE_TENSORS}" || "${prefix}" == "llama3" ]] || ddp+=(--write_tensors 0)
  run_ddp_case "${prefix}_3 (fp32, 8 GPUs)" "${program}" "${input_bin}" "${common[@]}" "${ddp[@]}" --dtype float32
  run_ddp_case "${prefix}_3_bfloat16 (8 GPUs)" "${program}" "${input_bin}" "${common[@]}" "${ddp[@]}" --dtype bfloat16
}

{
  if (( RUN_GPT2 )); then
    run_model_cases gpt2 "${TRAINING_DIR}/train_gpt2.py" "${GPT2_INPUT_BIN}"
  fi
  if (( RUN_LLAMA3 )); then
    run_model_cases llama3 "${SCRIPT_DIR}/train_llama3.2_1B.py" "${LLAMA3_INPUT_BIN}"
  fi

  echo "=========================================="
  echo "All Maca training jobs finished."
  echo "=========================================="
} 2>&1 | tee "${RESULT_LOG}"
