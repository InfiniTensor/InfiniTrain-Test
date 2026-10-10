# Qwen3 Megatron baseline

This shadow baseline loads the same complete Qwen3 LLMC checkpoint and token stream as InfiniTrain. The adapter parses the LLMC v4 layout, loads Q/K norm weights, repacks GQA QKV into Megatron order, and slices TP shards in memory. Pipeline stages are mapped to global LLMC layers with Megatron layer numbers.

## Prepare

Convert a Hugging Face checkpoint once and reuse the resulting LLMC file by both backends:

```bash
python baseline/megatron/models/qwen3/prepare/convert_hf_qwen3_to_llmc.py --hf-path /data1/shared/jiyiming/models/Qwen3-8B --output /data1/shared/jiyiming/models/Qwen3-8B/qwen3-8b-fp32.llmc
```

Convert the ordered LLMC token stream to Megatron indexed dataset format:

```bash
QWEN3_INPUT_BIN=/path/to/tiny_shakespeare_train.bin bash baseline/megatron/models/qwen3/prepare/prepare_dataset.sh
```

For an InfiniTrain smoke comparison, `QWEN3_INPUT_BIN` must point to exactly the same token file passed to `qwen3 --input_bin`.

## Aligned cases

The case names and output log basenames intentionally match `InfiniTrain/scripts/test_config.json`:

| Case | Dtype | DP | TP | PP | SP | Micro batch | Tokens/step |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen3_tp8_fp32` | FP32 | 1 | 8 | 1 | off | 1 | 64 |
| `qwen3_tp8_bfloat16` | BF16 autocast | 1 | 8 | 1 | off | 1 | 64 |
| `qwen3_tp8_sp_fp32` | FP32 | 1 | 8 | 1 | on | 1 | 64 |
| `qwen3_pp8_fp32` | FP32 | 1 | 1 | 8 | off | 1 | 64 |

BF16 uses `--autocast-bfloat16`, keeping parameters FP32 while autocasting forward operators. This is the aligned semantic counterpart of InfiniTrain BF16 training. Native Megatron BF16 parameters remain available with `DTYPE=bfloat16` for performance experiments, but their loss trajectory is not expected to match.

Run all aligned smoke cases (two iterations each by default):

```bash
QWEN3_LLMC_FILEPATH=/data1/shared/jiyiming/models/Qwen3-8B/qwen3-8b-fp32.llmc bash baseline/megatron/models/qwen3/train/run_qwen3_cases.sh
```

Select cases or run a longer performance measurement:

```bash
CASES=qwen3_tp8_fp32,qwen3_pp8_fp32 TRAIN_ITERS=10 bash baseline/megatron/models/qwen3/train/run_qwen3_cases.sh
```

The SP case uses Transformer Engine because the local RMSNorm implementation does not implement sequence parallelism. Its FP32 loss comparison therefore uses a separate 2e-2 absolute threshold; the observed Qwen3 smoke difference is 1.59e-2. Step timing is synchronized across ranks and excludes step 1 from the summary average.

PP8 uses Megatron's explicit layout `Et*5|t*5|t*5|t*5|t*4|t*4|t*4|t*4L`, matching InfiniTrain's 36-layer split of 5,5,5,5,4,4,4,4.

## Compare

Given an InfiniTrain log directory containing matching `qwen3_*.log` names:

```bash
python baseline/megatron/scripts/compare_loss.py /path/to/infinitrain/logs/qwen3 baseline/megatron/artifacts/qwen3/logs
```

Throughput uses the same filename pairing:

```bash
python baseline/megatron/scripts/compare_tps.py /path/to/infinitrain/logs/qwen3 baseline/megatron/artifacts/qwen3/logs
```

For an individual pair, use `models/qwen3/compare/compare_loss.sh`. Megatron iteration-2 running-average loss is corrected by default. The two-iteration matrix is a correctness smoke; throughput conclusions require an idle machine and at least ten measured iterations.

The current Megatron adapter does not yet provide an aligned LoRA case. Do not map `qwen3_tp8_lora_bfloat16` to a native Megatron BF16 run: InfiniTrain keeps frozen parameters in FP32 and trains only LoRA parameters, so the semantics and loss trajectory are different.
