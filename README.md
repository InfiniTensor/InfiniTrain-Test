# InfiniTrain-Test

InfiniTrain-Test stores reference logs and framework baselines used to validate
InfiniTrain training accuracy and performance.

## Repository layout

```text
.
├── baseline/
│   ├── pytorch/                 # PyTorch reference implementations and cases
│   └── megatron/
│       ├── common/              # shared Megatron adapters and comparison tools
│       ├── models/
│       │   └── llama3/          # Llama3 data preparation, training, and validation
│       ├── scripts/             # batch loss and throughput comparison tools
│       └── artifacts/           # generated data, caches, logs, and results (ignored)
├── infinitrain/
│   ├── logs/                    # archived InfiniTrain results by platform and date
│   └── scripts/                 # result collection and reporting utilities
└── third_party/
    └── Megatron-LM/             # Megatron-LM submodule
```

Generated files under `baseline/megatron/artifacts/` are not committed.

## Llama3 Megatron baseline

Prepare the Megatron dataset and run the six FP32/BF16 basic cases:

```bash
bash baseline/megatron/models/llama3/prepare/prepare_dataset.sh
bash baseline/megatron/models/llama3/train/run_basic_cases.sh
```

Compare an InfiniTrain run directory with the generated Megatron logs:

```bash
python3 baseline/megatron/scripts/compare_loss.py /path/to/infinitrain/logs/basic baseline/megatron/artifacts/llama3/logs --include-prefix llama3_
python3 baseline/megatron/scripts/compare_tps.py /path/to/infinitrain/logs/basic baseline/megatron/artifacts/llama3/logs --include-prefix llama3_
```

See `baseline/megatron/models/llama3/README.md` for the case matrix,
precision semantics, individual-case diagnostics, and configuration overrides.
