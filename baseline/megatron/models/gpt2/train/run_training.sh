#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
MEGATRON_BASELINE="${REPO_ROOT}/baseline/megatron"
MEGATRON_PATH="${MEGATRON_PATH:-${REPO_ROOT}/third_party/Megatron-LM}"
DATA_PREFIX="${GPT2_MEGATRON_DATA_PREFIX:-${MEGATRON_BASELINE}/artifacts/gpt2/datasets/tiny_shakespeare_llmc_text_document}"
LLMC_FILEPATH="${GPT2_LLMC_FILEPATH:-/data1/shared/InfiniTrain-dev/data/llmc/gpt2/gpt2_124M.bin}"
RESULT_LOG="${RESULT_LOG:-${MEGATRON_BASELINE}/artifacts/gpt2/logs/result_megatron_training.log}"
CACHE_PATH="${GPT2_MEGATRON_CACHE_PATH:-${MEGATRON_BASELINE}/artifacts/gpt2/cache}"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
TP="${TP:-1}"
PP="${PP:-1}"
SP="${SP:-0}"
VPP="${VPP:-1}"
TRANSFORMER_IMPL="${TRANSFORMER_IMPL:-local}"
PIPELINE_LAYOUT="${PIPELINE_LAYOUT:-}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
TOTAL_BATCH_TOKENS="${TOTAL_BATCH_TOKENS:-256}"
SEQ_LENGTH="${SEQ_LENGTH:-64}"
TRAIN_ITERS="${TRAIN_ITERS:-10}"
LR="${LR:-1e-4}"
DTYPE="${DTYPE:-float32}"
LOG_STEP_PERFORMANCE="${LOG_STEP_PERFORMANCE:-0}"

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
if [[ "${DTYPE}" == "float32" ]]; then
  export GPT2_DISABLE_TF32="${GPT2_DISABLE_TF32:-1}"
fi

WORLD_SIZE="$((NPROC_PER_NODE))"
if (( WORLD_SIZE % (TP * PP) != 0 )); then
  echo "world size ${WORLD_SIZE} is not divisible by TP*PP=$((TP * PP))" >&2
  exit 2
fi
DP="$((WORLD_SIZE / (TP * PP)))"
if (( TOTAL_BATCH_TOKENS % SEQ_LENGTH != 0 )); then
  echo "TOTAL_BATCH_TOKENS must be divisible by SEQ_LENGTH" >&2
  exit 2
fi
GLOBAL_BATCH_SIZE="$((TOTAL_BATCH_TOKENS / SEQ_LENGTH))"
if (( GLOBAL_BATCH_SIZE % (MICRO_BATCH_SIZE * DP) != 0 )); then
  echo "global sample batch ${GLOBAL_BATCH_SIZE} must be divisible by micro batch * DP=$((MICRO_BATCH_SIZE * DP))" >&2
  exit 2
fi

DTYPE_ARGS=()
case "${DTYPE}" in
  float32) ;;
  bfloat16) DTYPE_ARGS+=(--bf16) ;;
  autocast_bfloat16) DTYPE_ARGS+=(--autocast-bfloat16) ;;
  *) echo "unsupported DTYPE=${DTYPE}" >&2; exit 2 ;;
esac
PERFORMANCE_ARGS=()
if [[ "${LOG_STEP_PERFORMANCE}" == "1" ]]; then PERFORMANCE_ARGS+=(--log-step-performance); fi
PARALLEL_ARGS=()
if (( PP > 1 )); then PARALLEL_ARGS+=(--untie-embeddings-and-output-weights); fi
if [[ "${SP}" == "1" ]]; then PARALLEL_ARGS+=(--sequence-parallel); fi
if [[ -n "${PIPELINE_LAYOUT}" ]]; then
  PARALLEL_ARGS+=(--pipeline-model-parallel-layout "${PIPELINE_LAYOUT}")
elif (( VPP > 1 )); then
  if (( PP <= 1 )); then echo "VPP requires PP > 1" >&2; exit 2; fi
  if (( 12 % (PP * VPP) != 0 )); then echo "num layers must be divisible by PP*VPP or PIPELINE_LAYOUT must be set" >&2; exit 2; fi
  PARALLEL_ARGS+=(--num-layers-per-virtual-pipeline-stage "$((12 / (PP * VPP)))")
fi
VOCAB_DIVISOR=1
if (( TP > 1 )); then VOCAB_DIVISOR="$((48 / TP))"; fi

mkdir -p "$(dirname "${RESULT_LOG}")" "${CACHE_PATH}"

ARGS=(
  --use-mcore-models
  --transformer-impl "${TRANSFORMER_IMPL}"
  --attention-backend unfused
  --num-layers 12
  --hidden-size 768
  --num-attention-heads 12
  --group-query-attention
  --num-query-groups 12
  --seq-length "${SEQ_LENGTH}"
  --max-position-embeddings 1024
  --position-embedding-type learned_absolute
  --openai-gelu
  --normalization LayerNorm
  --norm-epsilon 1e-5
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --micro-batch-size "${MICRO_BATCH_SIZE}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --train-iters "${TRAIN_ITERS}"
  --lr "${LR}"
  --min-lr "${LR}"
  --lr-decay-style constant
  --optimizer sgd
  --sgd-momentum 0.0
  --weight-decay 0.0
  --clip-grad 0.0
  --tensor-model-parallel-size "${TP}"
  --pipeline-model-parallel-size "${PP}"
  --data-path "${DATA_PREFIX}"
  --data-cache-path "${CACHE_PATH}"
  --split 100,0,0
  --dataloader-type single
  --num-workers 0
  --tokenizer-type NullTokenizer
  --vocab-size 50257
  --make-vocab-size-divisible-by "${VOCAB_DIVISOR}"
  --seed 42
  --no-bias-gelu-fusion
  --no-bias-dropout-fusion
  --no-masked-softmax-fusion
  --no-gradient-accumulation-fusion
  --llmc-filepath "${LLMC_FILEPATH}"
  --log-interval 1
  --eval-interval "${TRAIN_ITERS}"
  --eval-iters 0
  --exit-interval "${TRAIN_ITERS}"
  "${DTYPE_ARGS[@]}"
  "${PERFORMANCE_ARGS[@]}"
  "${PARALLEL_ARGS[@]}"
)

TEE_ARGS=()
if [[ "${APPEND_RESULT_LOG:-0}" == "1" ]]; then TEE_ARGS+=(-a); fi
{
  echo "InfiniTrain-Test commit: $(git -C "${REPO_ROOT}" rev-parse HEAD)"
  echo "Megatron-LM commit: $(git -C "${MEGATRON_PATH}" rev-parse HEAD)"
  echo "DP=${DP} TP=${TP} PP=${PP} SP=${SP} VPP=${VPP} transformer_impl=${TRANSFORMER_IMPL} dtype=${DTYPE} micro_batch=${MICRO_BATCH_SIZE} global_batch=${GLOBAL_BATCH_SIZE} seq=${SEQ_LENGTH}"
  PYTHONPATH="${MEGATRON_PATH}:${PYTHONPATH:-}" \
    torchrun --standalone --nproc_per_node "${NPROC_PER_NODE}" \
    "${SCRIPT_DIR}/pretrain.py" "${ARGS[@]}"
} 2>&1 | tee "${TEE_ARGS[@]}" "${RESULT_LOG}"
