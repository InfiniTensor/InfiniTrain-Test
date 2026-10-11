#!/usr/bin/env python3
"""Use upstream pretrain_gpt/GPTDataset with an InfiniTrain LLMC checkpoint."""

import pathlib
import os
import struct
import sys
import time

import numpy as np
import torch

if os.environ.get("GPT2_DISABLE_TF32") == "1":
    torch.backends.cuda.matmul.fp32_precision = "ieee"

try:
    import nvidia_resiliency_ext as nvrx
    if not hasattr(nvrx, "__version__"):
        nvrx.__version__ = "0.6.0"
except ModuleNotFoundError:
    pass

MEGATRON_PATH = pathlib.Path(os.environ.get(
    "MEGATRON_PATH",
    pathlib.Path(__file__).resolve().parents[5] / "third_party" / "Megatron-LM",
))
sys.path.insert(0, str(MEGATRON_PATH))

import pretrain_gpt as upstream
from megatron.core import parallel_state
from megatron.core import tensor_parallel
from megatron.core.datasets import gpt_dataset
from gpt_builders import gpt_builder
from megatron.core.enums import ModelType
from megatron.training import get_args, pretrain
from megatron.training import training as megatron_training
from megatron.training.argument_utils import pretrain_cfg_container_from_args
from megatron.training.arguments import parse_and_validate_args
from model_provider import model_provider

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

def read_tensor(stream, shape):
    count = int(np.prod(shape))
    raw = stream.read(count * 4)
    if len(raw) != count * 4:
        raise RuntimeError(f"short LLMC tensor {shape}")
    return torch.from_numpy(np.frombuffer(raw, dtype=np.float32).copy().reshape(shape))

def assign(target, source, name):
    if tuple(target.shape) != tuple(source.shape):
        raise RuntimeError(f"{name}: {tuple(target.shape)} != {tuple(source.shape)}")
    target.copy_(source.to(target.device, target.dtype))

def assign_norm(norm, fused_linear, source, name, parameter):
    target = getattr(norm, parameter, None)
    if target is None:
        target = getattr(fused_linear, f"layer_norm_{parameter}", None)
    if target is None:
        raise RuntimeError(f"{name}: normalization {parameter} not found")
    assign(target, source, name)

def tensor_parallel_slice(value, rank, size, axis):
    if value.shape[axis] % size:
        raise RuntimeError(f"cannot shard {tuple(value.shape)} axis {axis} over TP={size}")
    width = value.shape[axis] // size
    return value.narrow(axis, rank * width, width)

def load_llmc(model, path):
    args = get_args()
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    with open(path, "rb") as stream, torch.no_grad():
        header = struct.unpack("<256i", stream.read(1024))
        if header[:8] != (20240326, 3, 1024, 50257, 12, 12, 768, 50304):
            raise RuntimeError(f"unexpected LLMC header: {header[:8]}")
        padded_wte = read_tensor(stream, (50304, 768))
        wpe = read_tensor(stream, (1024, 768))
        shapes = {
            "ln1w": (768,), "ln1b": (768,), "qkvw": (2304, 768), "qkvb": (2304,),
            "projw": (768, 768), "projb": (768,), "ln2w": (768,), "ln2b": (768,),
            "fc1w": (3072, 768), "fc1b": (3072,), "fc2w": (768, 3072), "fc2b": (768,),
        }
        values = {name: [read_tensor(stream, shape) for _ in range(12)] for name, shape in shapes.items()}
        final_w, final_b = read_tensor(stream, (768,)), read_tensor(stream, (768,))
        if stream.read(1):
            raise RuntimeError("trailing LLMC checkpoint data")

        embedding = tensor_parallel_slice(padded_wte if tp_size > 1 else padded_wte[:50257], tp_rank, tp_size, 0)
        if model.pre_process:
            assign(model.embedding.word_embeddings.weight, embedding, "wte")
            assign(model.embedding.position_embeddings.weight, wpe, "wpe")
        for layer in model.decoder.layers:
            i = layer.layer_number - 1
            qkvw = values["qkvw"][i].view(3, 12, 64, 768).permute(1, 0, 2, 3)
            qkvb = values["qkvb"][i].view(3, 12, 64).permute(1, 0, 2)
            qkvw = tensor_parallel_slice(qkvw, tp_rank, tp_size, 0).reshape(-1, 768)
            qkvb = tensor_parallel_slice(qkvb, tp_rank, tp_size, 0).reshape(-1)
            assign_norm(layer.input_layernorm, layer.self_attention.linear_qkv, values["ln1w"][i], f"layer{i}.ln1w", "weight")
            assign_norm(layer.input_layernorm, layer.self_attention.linear_qkv, values["ln1b"][i], f"layer{i}.ln1b", "bias")
            assign(layer.self_attention.linear_qkv.weight, qkvw, f"layer{i}.qkvw")
            assign(layer.self_attention.linear_qkv.bias, qkvb, f"layer{i}.qkvb")
            assign(layer.self_attention.linear_proj.weight, tensor_parallel_slice(values["projw"][i], tp_rank, tp_size, 1), f"layer{i}.projw")
            assign(layer.self_attention.linear_proj.bias, values["projb"][i], f"layer{i}.projb")
            assign_norm(layer.pre_mlp_layernorm, layer.mlp.linear_fc1, values["ln2w"][i], f"layer{i}.ln2w", "weight")
            assign_norm(layer.pre_mlp_layernorm, layer.mlp.linear_fc1, values["ln2b"][i], f"layer{i}.ln2b", "bias")
            assign(layer.mlp.linear_fc1.weight, tensor_parallel_slice(values["fc1w"][i], tp_rank, tp_size, 0), f"layer{i}.fc1w")
            assign(layer.mlp.linear_fc1.bias, tensor_parallel_slice(values["fc1b"][i], tp_rank, tp_size, 0), f"layer{i}.fc1b")
            assign(layer.mlp.linear_fc2.weight, tensor_parallel_slice(values["fc2w"][i], tp_rank, tp_size, 1), f"layer{i}.fc2w")
            assign(layer.mlp.linear_fc2.bias, values["fc2b"][i], f"layer{i}.fc2b")
        if model.post_process:
            assign(model.shared_embedding_or_output_weight() if model.share_embeddings_and_output_weights else model.output_layer.weight, embedding, "output")
            assign(model.decoder.final_layernorm.weight, final_w, "lnf.weight")
            assign(model.decoder.final_layernorm.bias, final_b, "lnf.bias")
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    print(f"loaded LLMC checkpoint on TP rank {tp_rank}/{tp_size}, "
          f"PP rank {pp_rank}/{args.pipeline_model_parallel_size}: {path}", flush=True)

def aligned_vocab_loss(self, labels, logits):
    args = get_args()
    partition_size = logits.size(-1)
    vocab_start = parallel_state.get_tensor_model_parallel_rank() * partition_size
    valid_size = max(0, min(partition_size, args.vocab_size - vocab_start))
    if valid_size < partition_size:
        logits = logits.clone()
        logits[..., valid_size:] = float("-inf")
    labels = labels.transpose(0, 1).contiguous()
    loss = tensor_parallel.vocab_parallel_cross_entropy(logits, labels)
    return loss.transpose(0, 1).contiguous()


def aligned_gpt_builder(*args, **kwargs):
    model = gpt_builder(*args, **kwargs)
    if model.post_process and parallel_state.get_tensor_model_parallel_world_size() > 1:
        model.compute_language_model_loss = aligned_vocab_loss.__get__(model, type(model))
    return model


def llmc_model_provider(pre_process=True, post_process=True, vp_stage=None, config=None, pg_collection=None):
    model = model_provider(aligned_gpt_builder, pre_process, post_process, vp_stage, config, pg_collection)
    load_llmc(model, get_args().llmc_filepath)
    return model

if __name__ == "__main__":
    gpt_dataset._build_shuffle_index = ordered_shuffle_index
    print("GPTDataset shuffle index disabled for InfiniTrain alignment", flush=True)
    setattr(upstream.train_valid_test_datasets_provider, "is_distributed", True)
    args = parse_and_validate_args(extra_args_provider=add_args, args_defaults={"tokenizer_type": "NullTokenizer"})
    if args.log_step_performance:
        install_performance_logger()
    forward_step = autocast_forward_step if args.autocast_bfloat16 else upstream.forward_step
    pretrain(
        pretrain_cfg_container_from_args(args), upstream.train_valid_test_datasets_provider,
        llmc_model_provider, ModelType.encoder_or_decoder, forward_step,
        get_embedding_ranks=upstream.get_embedding_ranks,
    )
