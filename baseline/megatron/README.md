# Llama3 Megatron baseline

This shadow baseline compares InfiniTrain and Megatron-LM using the same Llama3
LLMC FP32 checkpoint and the same ordered Tiny Shakespeare token stream. It does
not replace or modify the existing PyTorch Llama3 baseline.

## Prepare and run

```bash
bash baseline/megatron/scripts/llama3/prepare.sh

DTYPE=float32 NPROC_PER_NODE=1 TP=1 PP=1 bash baseline/megatron/scripts/llama3/run.sh
```

Run the sixteen basic FP32/BF16 cases. Cases 1-3 cover single-GPU and DDP;
cases 4-8 add TP, SP, PP, VPP, and the core DP+TP+PP combination.

```bash
bash baseline/megatron/scripts/llama3/run_cases.sh

# Run only selected cases.
CASES=llama3_1,llama3_3_bfloat16 bash baseline/megatron/scripts/llama3/run_cases.sh
```

The aligned basic matrix is:

| Case | Dtype | GPUs | DP | TP | PP | SP | VPP | Micro batch | Global batch | Tokens/step | Seq | Impl |
| --- | --- | ---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: | --- |
| `llama3_1` | FP32 | 1 | 1 | 1 | 1 | off | 1 | 4 | 4 | 256 | 64 | local |
| `llama3_1_bfloat16` | BF16 autocast | 1 | 1 | 1 | 1 | off | 1 | 4 | 4 | 256 | 64 | local |
| `llama3_2` | FP32 | 1 | 1 | 1 | 1 | off | 1 | 80 | 80 | 5120 | 64 | local |
| `llama3_2_bfloat16` | BF16 autocast | 1 | 1 | 1 | 1 | off | 1 | 80 | 80 | 5120 | 64 | local |
| `llama3_3` | FP32 | 8 | 8 | 1 | 1 | off | 1 | 10 | 80 | 5120 | 64 | local |
| `llama3_3_bfloat16` | BF16 autocast | 8 | 8 | 1 | 1 | off | 1 | 10 | 80 | 5120 | 64 | local |
| `llama3_4` | FP32 | 8 | 2 | 4 | 1 | off | 1 | 40 | 80 | 5120 | 64 | local |
| `llama3_4_bfloat16` | BF16 autocast | 8 | 2 | 4 | 1 | off | 1 | 40 | 80 | 5120 | 64 | local |
| `llama3_5` | FP32 | 8 | 2 | 4 | 1 | on | 1 | 40 | 80 | 5120 | 64 | TE |
| `llama3_5_bfloat16` | BF16 autocast | 8 | 2 | 4 | 1 | on | 1 | 40 | 80 | 5120 | 64 | TE |
| `llama3_6` | FP32 | 8 | 1 | 1 | 8 | off | 1 | 10 | 80 | 5120 | 64 | local |
| `llama3_6_bfloat16` | BF16 autocast | 8 | 1 | 1 | 8 | off | 1 | 10 | 80 | 5120 | 64 | local |
| `llama3_7` | FP32 | 4 | 1 | 1 | 4 | off | 2 | 10 | 80 | 5120 | 64 | local |
| `llama3_7_bfloat16` | BF16 autocast | 4 | 1 | 1 | 4 | off | 2 | 10 | 80 | 5120 | 64 | local |
| `llama3_8` | FP32 | 8 | 2 | 2 | 2 | off | 1 | 40 | 80 | 5120 | 64 | local |
| `llama3_8_bfloat16` | BF16 autocast | 8 | 2 | 2 | 2 | off | 1 | 40 | 80 | 5120 | 64 | local |

Case 5 uses Transformer Engine because the local Torch RMSNorm does not support
sequence parallelism. Case 8 intentionally covers DP+TP+PP without SP/VPP:
its aligned batch has one microbatch (`80 / (40 * DP2)`), while Megatron VPP2
requires at least two. Case 7 exercises VPP2 separately with eight microbatches.

The BF16 basic cases use `DTYPE=autocast_bfloat16` to match the
InfiniTrain/PyTorch precision policy: parameters remain FP32 while the forward
pass runs under PyTorch BF16 autocast. `DTYPE=bfloat16` remains available for
native Megatron BF16 runs, where model parameters are stored in BF16; its loss
trajectory is therefore not expected to match the aligned BF16 cases exactly.

The basic cases enable synchronized step-performance logging by default. The
timed region is Megatron's standard training step, including data loading,
forward, loss, backward, gradient communication, and the optimizer update. On
multiple GPUs the reported latency is the slowest rank. Step 1 is treated as
warmup, and the summary averages steps 2 through 10, matching the PyTorch
baseline. Disable it with `LOG_STEP_PERFORMANCE=0`, or enable it for an
individual run with `LOG_STEP_PERFORMANCE=1`.

Each run also prints a stable configuration header and PyTorch-compatible summary:

```text
Framework: Megatron-LM
Model: llama3
Config: GPUs=... DP=... TP=... PP=... dtype=... micro_batch=... global_batch=... seq=... steps=... lr=...
final 9 iters Latency avg: ...ms
final 9 iters Throughput avg: ... tok/s
peak memory consumption: ... MiB
```

For distributed runs, latency and peak memory use the maximum value across ranks.
`run_cases.sh` writes the filtered key output for all selected cases to
`baseline/megatron/logs/qy_a100_g3025/result_megatron_training_llama3.log`.
The complete output for each case remains available under
`baseline/megatron/artifacts/llama3/logs/` for diagnostics.

InfiniTrain uses the existing `llama3` executable with the same defaults:

```bash
../InfiniTrain/build/llama3 \
  --input_bin=/data1/shared/InfiniTrain-dev/data/llmc/llama3/tinyshakespeare/tiny_shakespeare_train.bin \
  --llmc_filepath=/data1/shared/InfiniTrain-dev/data/llmc/llama3/llama3.2_1B_fp32.bin \
  --dtype=float32 --batch_size=4 --sequence_length=64 \
  --total_batch_size=256 --num_iteration=10 --learning_rate=1e-5 \
  --tensor_parallel=1 --pipeline_parallel=1
```

Loss and throughput comparisons use the repository-level tools shared by all
frameworks and models. To compare one case, pass it through `--include-cases`:

```bash
python tools/compare/compare_loss.py \
  /path/to/infinitrain/logs baseline/megatron/artifacts/llama3/logs \
  --include-cases llama3_1
```

For batch regression summaries, pass the completed flat log directory directly.
The scripts automatically pair `infinitrain_*.log` with the corresponding
Megatron log and report loss and throughput separately:

```bash
python tools/compare/compare_loss.py baseline/megatron/artifacts/llama3/logs
```

```bash
python tools/compare/compare_tps.py baseline/megatron/artifacts/llama3/logs
```

Both scripts recursively collect `.log` files and print missing-file and total-case summaries.
Megatron emits the same `step N/M | train loss ... | (... tok/s)` records as
InfiniTrain and PyTorch, so the loss and TPS tools share one parser without
framework-specific loss correction. TPS reports each backend's average
throughput and the InfiniTrain/Megatron percentage without a performance threshold.
The standard InfiniTrain `run_models_and_profile.bash` output can be passed as
the first directory. Use `--include-prefix llama3_` when its `basic/` directory
also contains other models:

```bash
python tools/compare/compare_loss.py /path/to/infinitrain/run/logs/basic baseline/megatron/artifacts/llama3/logs --include-prefix llama3_
```

```bash
python tools/compare/compare_tps.py /path/to/infinitrain/run/logs/basic baseline/megatron/artifacts/llama3/logs --include-prefix llama3_
```

The adapter parses the LLMC header, validates the full file size, repacks block
Q/K/V into Megatron GQA order, and packs SwiGLU as `[c_fc2 gate, c_fc up]`.
For pipeline parallel runs it loads embeddings only on `pre_process` stages, final
normalization/output weights only on `post_process` stages, and maps each local
layer to its global LLMC index using Megatron's `layer_number`.
