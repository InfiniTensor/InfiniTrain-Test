# Qwen3 Megatron baseline

This shadow baseline loads the same complete Qwen3 LLMC checkpoint and token
stream as InfiniTrain. The adapter performs TP slicing and Megatron QKV layout
conversion in memory.

## Prepare data

```bash
QWEN3_INPUT_BIN=/path/to/llmc_tokens.bin \
  bash baseline/megatron/models/qwen3/prepare/prepare_dataset.sh
```

Create the complete LLMC v4 checkpoint shared by InfiniTrain and Megatron:

```bash
python baseline/megatron/models/qwen3/prepare/convert_hf_qwen3_to_llmc.py \
  --hf-path /path/to/Qwen3-8B \
  --output /path/to/qwen3-8b-fp32.llmc
```

The converter writes `magic=20240804`, `version=4`, and stores
`intermediate_size` in header slot 8. Keep this contract synchronized with the
InfiniTrain Qwen3 checkpoint loader.
## Run

```bash
QWEN3_LLMC_FILEPATH=/path/to/qwen3-8b-fp32.llmc \
DTYPE=float32 NPROC_PER_NODE=8 TP=8 PP=1 \
  bash baseline/megatron/models/qwen3/train/run_training.sh
```

## Compare

```bash
INFINITRAIN_LOG=/path/to/infinitrain.log \
  bash baseline/megatron/models/qwen3/compare/compare_loss.sh
```
