# Llama3 Megatron baseline

This shadow baseline compares InfiniTrain and Megatron-LM using the same Llama3
LLMC FP32 checkpoint and the same ordered Tiny Shakespeare token stream. It does
not replace or modify the existing PyTorch Llama3 baseline.

## Prepare and run

```bash
bash baseline/megatron/models/llama3/prepare/prepare_dataset.sh

DTYPE=float32 NPROC_PER_NODE=1 TP=1 PP=1 \
  bash baseline/megatron/models/llama3/train/run_training.sh
```

InfiniTrain uses the existing `llama3` executable with the same defaults:

```bash
../InfiniTrain/build/llama3 \
  --input_bin=/data1/shared/InfiniTrain-dev/data/llmc/llama3/tinyshakespeare/tiny_shakespeare_train.bin \
  --llmc_filepath=/data1/shared/InfiniTrain-dev/data/llmc/llama3/llama3.2_1B_fp32.bin \
  --dtype=float32 --batch_size=4 --sequence_length=64 \
  --total_batch_size=256 --num_iteration=10 --learning_rate=1e-5 \
  --tensor_parallel=1 --pipeline_parallel=1
```

Compare logs with:

```bash
INFINITRAIN_LOG=/path/to/infinitrain.log \
  bash baseline/megatron/models/llama3/compare/compare_loss.sh
```

The adapter parses the LLMC header, validates the full file size, repacks block
Q/K/V into Megatron GQA order, and packs SwiGLU as `[c_fc2 gate, c_fc up]`.
