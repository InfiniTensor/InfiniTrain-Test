#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
D=/mnt/afs/user_home/wangyichen/infinitrain-baseline
cd "${SCRIPT_DIR}"
mkdir -p "${D}/megatron" "${D}/logs"
EXPORT="${D}/megatron/mixtral_megatron_export.bin"

echo "=========================================="
echo "Exporting init model (num_iterations=0)"
echo "=========================================="
python train_megatron_mixtral_mlu.py \
  --write_model_path "${EXPORT}" \
  --megatron_path "" \
  --num_iterations 0 \
  --master_port 29571

run_case() {
  local name="$1" dtype="$2" mbs="$3" gbs="$4" port="$5"
  echo "=========================================="
  echo "Running: mixtral_${name} (${dtype})"
  echo "=========================================="
  python train_megatron_mixtral_mlu.py \
    --weights_path "${EXPORT}" \
    --input_bin "${D}/data/tinyshakespeare-llama3/tiny_shakespeare_train.bin" \
    --dtype "${dtype}" \
    --micro_batch_size "${mbs}" \
    --global_batch_size "${gbs}" \
    --num_iterations 10 \
    --megatron_path "" \
    --master_port "${port}" \
    --log_interval 1 \
    --print_timing
}

run_case "1" "float32" 4 4 29572
run_case "1_bfloat16" "bfloat16" 4 4 29573
run_case "2" "float32" 80 80 29574
run_case "2_bfloat16" "bfloat16" 80 80 29575

echo "=========================================="
echo "All Megatron MLU jobs finished."
echo "=========================================="
