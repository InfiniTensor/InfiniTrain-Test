#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
BASELINE="${REPO_ROOT}/baseline/megatron"
MEGATRON_PATH="${MEGATRON_PATH:-${REPO_ROOT}/third_party/Megatron-LM}"
DATA_PREFIX="${LLAMA3_MEGATRON_DATA_PREFIX:-${BASELINE}/artifacts/llama3/datasets/llama3_text_document}"
LLMC_FILEPATH="${LLAMA3_LLMC_FILEPATH:-/data1/shared/InfiniTrain-dev/data/llmc/llama3/llama3.2_1B_fp32.bin}"
RESULT_LOG="${RESULT_LOG:-${BASELINE}/artifacts/llama3/logs/result_megatron_training.log}"
CACHE_PATH="${LLAMA3_MEGATRON_CACHE_PATH:-${BASELINE}/artifacts/llama3/cache}"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
TP="${TP:-1}"
PP="${PP:-1}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
TOTAL_BATCH_TOKENS="${TOTAL_BATCH_TOKENS:-256}"
SEQ_LENGTH="${SEQ_LENGTH:-64}"
TRAIN_ITERS="${TRAIN_ITERS:-10}"
LR="${LR:-1e-5}"
DTYPE="${DTYPE:-float32}"

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
if [[ "${DTYPE}" == "float32" ]]; then export LLAMA3_DISABLE_TF32="${LLAMA3_DISABLE_TF32:-1}"; fi
if (( PP != 1 )); then echo "Llama3 LLMC runtime loading currently supports PP=1" >&2; exit 2; fi
if (( NPROC_PER_NODE % (TP * PP) != 0 )); then echo "world size is not divisible by TP*PP" >&2; exit 2; fi
DP="$((NPROC_PER_NODE / (TP * PP)))"
if (( TOTAL_BATCH_TOKENS % SEQ_LENGTH != 0 )); then echo "TOTAL_BATCH_TOKENS must be divisible by SEQ_LENGTH" >&2; exit 2; fi
GLOBAL_BATCH_SIZE="$((TOTAL_BATCH_TOKENS / SEQ_LENGTH))"
if (( GLOBAL_BATCH_SIZE % (MICRO_BATCH_SIZE * DP) != 0 )); then echo "global batch must be divisible by micro batch * DP" >&2; exit 2; fi

DTYPE_ARGS=()
case "${DTYPE}" in float32) ;; bfloat16) DTYPE_ARGS+=(--bf16) ;; *) echo "unsupported DTYPE=${DTYPE}" >&2; exit 2;; esac

mkdir -p "$(dirname "${RESULT_LOG}")" "${CACHE_PATH}"
ARGS=(
  --use-mcore-models --transformer-impl local --attention-backend unfused
  --num-layers 16 --hidden-size 2048 --ffn-hidden-size 8192
  --num-attention-heads 32 --group-query-attention --num-query-groups 8 --kv-channels 64
  --seq-length "${SEQ_LENGTH}" --max-position-embeddings 8192
  --position-embedding-type rope --rotary-percent 1.0 --rotary-base 500000 --rotary-interleaved --no-rope-fusion
  --normalization RMSNorm --no-persist-layer-norm --norm-epsilon 1e-5
  --swiglu --disable-bias-linear --untie-embeddings-and-output-weights
  --attention-dropout 0.0 --hidden-dropout 0.0
  --micro-batch-size "${MICRO_BATCH_SIZE}" --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --train-iters "${TRAIN_ITERS}" --lr "${LR}" --min-lr "${LR}" --lr-decay-style constant
  --optimizer adam --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-8
  --weight-decay 0.0 --clip-grad 0.0
  --tensor-model-parallel-size "${TP}" --pipeline-model-parallel-size "${PP}"
  --data-path "${DATA_PREFIX}" --data-cache-path "${CACHE_PATH}" --split 100,0,0
  --dataloader-type single --num-workers 0 --tokenizer-type NullTokenizer
  --vocab-size 128256 --make-vocab-size-divisible-by 1 --seed 42
  --no-masked-softmax-fusion --no-bias-swiglu-fusion --no-gradient-accumulation-fusion
  --llmc-filepath "${LLMC_FILEPATH}" --log-interval 1
  --eval-interval "${TRAIN_ITERS}" --eval-iters 0 --exit-interval "${TRAIN_ITERS}"
  "${DTYPE_ARGS[@]}"
)
{
  echo "InfiniTrain-Test commit: $(git -C "${REPO_ROOT}" rev-parse HEAD)"
  echo "Megatron-LM commit: $(git -C "${MEGATRON_PATH}" rev-parse HEAD)"
  echo "DP=${DP} TP=${TP} PP=${PP} dtype=${DTYPE} micro_batch=${MICRO_BATCH_SIZE} global_batch=${GLOBAL_BATCH_SIZE} seq=${SEQ_LENGTH}"
  MEGATRON_PATH="${MEGATRON_PATH}" PYTHONPATH="${MEGATRON_PATH}:${PYTHONPATH:-}" \
    torchrun --standalone --nproc_per_node "${NPROC_PER_NODE}" "${SCRIPT_DIR}/pretrain.py" "${ARGS[@]}"
} 2>&1 | tee "${RESULT_LOG}"
