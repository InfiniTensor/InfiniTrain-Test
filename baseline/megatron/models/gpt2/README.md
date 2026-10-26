# GPT-2 Megatron baseline

This shadow baseline compares InfiniTrain and Megatron-LM using the same GPT-2
LLMC FP32 checkpoint and the same ordered Tiny Shakespeare token stream. It
uses upstream Megatron training and GPTDataset without modifying
`third_party/Megatron-LM`. Direct LLMC loading supports the TP, SP, PP, and VPP combinations in the basic matrix.

## Prepare and run

```bash
bash baseline/megatron/models/gpt2/prepare/prepare_dataset.sh
bash baseline/megatron/models/gpt2/train/run_basic_cases.sh
```

Run only selected cases with `CASES`:

```bash
CASES=gpt2_1,gpt2_3_bfloat16 bash baseline/megatron/models/gpt2/train/run_basic_cases.sh
```

The basic matrix matches `baseline/pytorch/scripts/run_pytorch_training.sh`:

| Case | Dtype | GPUs / DP | Micro batch | Tokens/step | Sequence | Iterations |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `gpt2_1` | FP32 | 1 | 4 | 256 | 64 | 10 |
| `gpt2_1_bfloat16` | BF16 autocast | 1 | 4 | 256 | 64 | 10 |
| `gpt2_2` | FP32 | 1 | 80 | 5120 | 64 | 10 |
| `gpt2_2_bfloat16` | BF16 autocast | 1 | 80 | 5120 | 64 | 10 |
| `gpt2_3` | FP32 | 8 | 10 | 5120 | 64 | 10 |
| `gpt2_3_bfloat16` | BF16 autocast | 8 | 10 | 5120 | 64 | 10 |
| `gpt2_4` | FP32 | 8 / DP=2, TP=4 | 40 | 5120 | 64 | 10 |
| `gpt2_4_bfloat16` | BF16 autocast | 8 / DP=2, TP=4 | 40 | 5120 | 64 | 10 |
| `gpt2_5` | FP32 | 8 / DP=2, TP=4, SP | 40 | 5120 | 64 | 10 |
| `gpt2_5_bfloat16` | BF16 autocast | 8 / DP=2, TP=4, SP, TE | 40 | 5120 | 64 | 10 |
| `gpt2_6` | FP32 | 8 / PP=8 | 10 | 5120 | 64 | 10 |
| `gpt2_6_bfloat16` | BF16 autocast | 8 / PP=8 | 10 | 5120 | 64 | 10 |
| `gpt2_7` | FP32 | 4 / PP=4, VPP=2 | 10 | 5120 | 64 | 10 |
| `gpt2_7_bfloat16` | BF16 autocast | 4 / PP=4, VPP=2 | 10 | 5120 | 64 | 10 |
| `gpt2_8` | FP32 | 8 / DP=2, TP=2, PP=2, SP, VPP=2 | 20 | 5120 | 64 | 10 |
| `gpt2_8_bfloat16` | BF16 autocast | 8 / DP=2, TP=2, PP=2, SP, VPP=2, TE | 20 | 5120 | 64 | 10 |

The BF16 basic cases use `DTYPE=autocast_bfloat16`: parameters remain FP32
while the forward pass runs under PyTorch BF16 autocast, matching the
InfiniTrain/PyTorch precision policy. Native Megatron BF16 remains available
as `DTYPE=bfloat16` for diagnostics but is not part of the default matrix.

FP32 runs disable TF32 by default to use IEEE matrix multiplication. Set
`GPT2_DISABLE_TF32=0` only for diagnostic comparison with PyTorch TF32.

## Performance measurement

Performance logging is enabled by default for the basic matrix. The timed
region is Megatron's standard training step, including data loading, forward,
loss, backward, gradient communication, and the optimizer update. Multi-GPU
latency uses the slowest rank. Step 1 is warmup, and the summary averages steps
2 through 10. Set `LOG_STEP_PERFORMANCE=0` to disable it.

## Compare

For individual-case diagnostics and machine-readable per-step JSON:

```bash
INFINITRAIN_LOG=/path/to/infinitrain.log MEGATRON_LOG=/path/to/megatron.log OUTPUT_JSON=/path/to/result.json ATOL=1e-2 bash baseline/megatron/models/gpt2/compare/compare_loss.sh
```

For batch loss and throughput comparison against an InfiniTrain standard run
directory:

```bash
python3 baseline/megatron/scripts/compare_loss.py /path/to/infinitrain/logs/basic baseline/megatron/artifacts/gpt2/logs --include-prefix gpt2_ --threshold-fp32 2e-3 --include-cases gpt2_1,gpt2_1_bfloat16,gpt2_2,gpt2_2_bfloat16,gpt2_3,gpt2_3_bfloat16,gpt2_4,gpt2_4_bfloat16,gpt2_5,gpt2_5_bfloat16,gpt2_6,gpt2_6_bfloat16,gpt2_7,gpt2_7_bfloat16,gpt2_8,gpt2_8_bfloat16
```

```bash
python3 baseline/megatron/scripts/compare_tps.py /path/to/infinitrain/logs/basic baseline/megatron/artifacts/gpt2/logs --include-prefix gpt2_ --include-cases gpt2_1,gpt2_1_bfloat16,gpt2_2,gpt2_2_bfloat16,gpt2_3,gpt2_3_bfloat16,gpt2_4,gpt2_4_bfloat16,gpt2_5,gpt2_5_bfloat16,gpt2_6,gpt2_6_bfloat16,gpt2_7,gpt2_7_bfloat16,gpt2_8,gpt2_8_bfloat16
```

The GPT-2 FP32 threshold is `2e-3`; BF16 uses the batch comparer's default
`1e-2`. Throughput excludes step 1 and reports both averages and the
InfiniTrain/Megatron percentage without applying a pass/fail threshold.
The 256-token BF16 case is numerically sensitive and can exceed `1e-2` in
both the PyTorch/InfiniTrain and Megatron/InfiniTrain comparisons. Such a
result remains an explicit failure rather than silently loosening the
threshold.

The adapter validates the LLMC header, restores the runtime vocabulary size,
interleaves QKV per attention head for MCore, and replaces only GPTDataset's
shuffle-index builder with an ordered index.
