#!/usr/bin/env python3
"""Run upstream Megatron GPT training from an InfiniTrain Llama3 LLMC checkpoint."""

import os
import pathlib
import sys
import time

import numpy as np
import torch

if os.environ.get("LLAMA3_DISABLE_TF32") == "1":
    torch.backends.cuda.matmul.fp32_precision = "ieee"

try:
    import nvidia_resiliency_ext as nvrx
    if not hasattr(nvrx, "__version__"):
        nvrx.__version__ = "0.6.0"
except ModuleNotFoundError:
    pass

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[4]
MEGATRON_PATH = pathlib.Path(os.environ.get("MEGATRON_PATH", REPO_ROOT / "third_party" / "Megatron-LM"))
sys.path.insert(0, str(MEGATRON_PATH))
sys.path.insert(0, str(SCRIPT_DIR.parent / "adapter"))

import pretrain_gpt as upstream
from gpt_builders import gpt_builder
from megatron.core import parallel_state
from megatron.core.datasets import gpt_dataset
from megatron.core.enums import ModelType
from megatron.training import get_args, pretrain
from megatron.training import training as megatron_training
from megatron.training.argument_utils import pretrain_cfg_container_from_args
from megatron.training.arguments import parse_and_validate_args
from model_provider import model_provider
from llmc_loader import Reader, pack_qkv_for_megatron, tensor_parallel_slice


def add_args(parser):
    parser.add_argument("--llmc-filepath", required=True)
    parser.add_argument("--autocast-bfloat16", action="store_true",
                        help="Keep FP32 parameters and autocast the forward pass to BF16")
    parser.add_argument("--log-step-performance", action="store_true",
                        help="Report synchronized end-to-end train-step latency and throughput")
    return parser


def install_performance_logger():
    original_train_step = megatron_training.train_step
    measured = []

    def timed_train_step(*args, **kwargs):
        runtime_args = get_args()
        iteration = kwargs.get("iteration")
        if iteration is None and len(args) > 7:
            iteration = args[7]
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = original_train_step(*args, **kwargs)
        torch.cuda.synchronize()
        local_elapsed = time.perf_counter() - started

        # Exclude the slowest-rank reduction itself from the measured interval.
        elapsed = torch.tensor(local_elapsed, dtype=torch.float64, device=torch.cuda.current_device())
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(elapsed, op=torch.distributed.ReduceOp.MAX)
        elapsed_seconds = elapsed.item()
        step = (iteration if iteration is not None else len(measured)) + 1
        tokens_per_second = runtime_args.global_batch_size * runtime_args.seq_length / elapsed_seconds
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if rank == 0:
            print(f"PERF step {step}/{runtime_args.train_iters} | "
                  f"{elapsed_seconds * 1000:.2f} ms | {tokens_per_second:.0f} tok/s", flush=True)

        if step > 1 and not result[1]:
            measured.append((elapsed_seconds, tokens_per_second))
        if step == runtime_args.train_iters and rank == 0:
            if measured:
                avg_latency_ms = sum(item[0] for item in measured) / len(measured) * 1000
                avg_throughput = sum(item[1] for item in measured) / len(measured)
                print(f"PERF summary | measured_steps {len(measured)} | "
                      f"avg {avg_latency_ms:.2f} ms | avg {avg_throughput:.0f} tok/s", flush=True)
            else:
                print("PERF summary | measured_steps 0", flush=True)
        return result

    megatron_training.train_step = timed_train_step


def autocast_forward_step(*args, **kwargs):
    # Keep parameters in FP32 while using PyTorch's BF16 autocast policy for forward operators.
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return upstream.forward_step(*args, **kwargs)


def ordered_shuffle_index(num_samples, total_size, numpy_random_state):
    del num_samples, numpy_random_state
    return np.arange(total_size, dtype=np.uint32)


def assign(target, source, name):
    source = np.asarray(source)
    if tuple(target.shape) != tuple(source.shape):
        raise RuntimeError(f"{name}: Megatron {tuple(target.shape)} != LLMC {tuple(source.shape)}")
    target.copy_(torch.from_numpy(source.copy()).to(device=target.device, dtype=target.dtype))


def load_llmc(model, path):
    args = get_args()
    if args.pipeline_model_parallel_size != 1:
        raise RuntimeError("Llama3 LLMC runtime loading currently supports PP=1")
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    reader, tensors = Reader(path), None
    cfg = reader.config
    expected = (args.num_layers, args.hidden_size, args.num_attention_heads, args.num_query_groups,
                args.ffn_hidden_size, args.padded_vocab_size)
    actual = (cfg.num_layers, cfg.hidden_size, cfg.num_attention_heads, cfg.num_query_groups,
              cfg.intermediate_size, cfg.vocab_size)
    if actual != expected:
        raise RuntimeError(f"Llama3 config mismatch: LLMC={actual}, Megatron={expected}")
    if cfg.use_scaled_rope:
        raise RuntimeError("this adapter does not yet implement Llama scaled RoPE")
    tensors = reader.tensors
    with torch.no_grad():
        assign(model.embedding.word_embeddings.weight, tensor_parallel_slice(tensors["embedding"], tp_rank, tp_size, 0), "embedding")
        for i, layer in enumerate(model.decoder.layers):
            assign(layer.input_layernorm.weight, tensors["input_norm"][i], f"layer{i}.input_norm")
            assign(layer.self_attention.linear_qkv.weight, pack_qkv_for_megatron(tensors["qkv"][i], cfg, tp_rank, tp_size), f"layer{i}.qkv")
            assign(layer.self_attention.linear_proj.weight, tensor_parallel_slice(tensors["attention_output"][i], tp_rank, tp_size, 1), f"layer{i}.attention_output")
            assign(layer.pre_mlp_layernorm.weight, tensors["pre_mlp_norm"][i], f"layer{i}.pre_mlp_norm")
            gate = tensor_parallel_slice(tensors["gate"][i], tp_rank, tp_size, 0)
            up = tensor_parallel_slice(tensors["up"][i], tp_rank, tp_size, 0)
            assign(layer.mlp.linear_fc1.weight, np.concatenate([gate, up], axis=0), f"layer{i}.linear_fc1")
            assign(layer.mlp.linear_fc2.weight, tensor_parallel_slice(tensors["down"][i], tp_rank, tp_size, 1), f"layer{i}.linear_fc2")
        assign(model.decoder.final_layernorm.weight, tensors["final_norm"], "final_norm")
        assign(model.output_layer.weight, tensor_parallel_slice(tensors["output"], tp_rank, tp_size, 0), "output")
    print(f"loaded Llama3 LLMC checkpoint on TP rank {tp_rank}/{tp_size}: {path}", flush=True)


def llmc_model_provider(pre_process=True, post_process=True, vp_stage=None, config=None, pg_collection=None):
    model = model_provider(gpt_builder, pre_process, post_process, vp_stage, config, pg_collection)
    load_llmc(model, get_args().llmc_filepath)
    return model


if __name__ == "__main__":
    gpt_dataset._build_shuffle_index = ordered_shuffle_index
    setattr(upstream.train_valid_test_datasets_provider, "is_distributed", True)
    args = parse_and_validate_args(extra_args_provider=add_args, args_defaults={"tokenizer_type": "NullTokenizer"})
    if args.log_step_performance:
        install_performance_logger()
    forward_step = autocast_forward_step if args.autocast_bfloat16 else upstream.forward_step
    pretrain(pretrain_cfg_container_from_args(args), upstream.train_valid_test_datasets_provider,
             llmc_model_provider, ModelType.encoder_or_decoder, forward_step,
             get_embedding_ranks=upstream.get_embedding_ranks)
