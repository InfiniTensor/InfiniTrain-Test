#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTORCH_BASELINE_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

export PYTHON="${PYTHON:-/opt/conda/bin/python}"
export TORCHRUN="${TORCHRUN:-/opt/conda/bin/torchrun}"
export HF_HOME="${HF_HOME:-/private/jiyiming/models/HuggingFace}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export GPT2_INPUT_BIN="${GPT2_INPUT_BIN:-/private/jiyiming/data/llmc/gpt2/tinyshakespeare/tiny_shakespeare_train.bin}"
export MACA_RUNTIME_ROOT="${MACA_RUNTIME_ROOT:-/opt/hpcc-3.5.3}"
export MACA_RUNTIME_WORKDIR="${MACA_RUNTIME_WORKDIR:-/opt/hpcc/lib}"
export RESULT_LOG="${RESULT_LOG:-${PYTORCH_BASELINE_DIR}/logs/maca_mars_x203/result_pytorch_training.log}"
export RUN_GPT2="${RUN_GPT2:-1}"
export RUN_LLAMA3="${RUN_LLAMA3:-0}"
export WRITE_TENSORS="${WRITE_TENSORS:-0}"
export TORCH_CPP_LOG_LEVEL="${TORCH_CPP_LOG_LEVEL:-ERROR}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

exec bash "${SCRIPT_DIR}/run_pytorch_training_maca.sh"
