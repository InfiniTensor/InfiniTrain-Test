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

def register_tensor_dumps(model):
    output_dir = os.environ.get("GPT2_TENSOR_DUMP_DIR")
    if not output_dir:
        return
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    counters = {}

    def save(name, value):
        index = counters.get(name, 0)
        counters[name] = index + 1
        suffix = f"_{index}" if index else ""
        np.save(
            output_dir / f"{name}{suffix}_forward.npy",
            value.detach().float().cpu().contiguous().numpy(),
        )

    def register(module, name):
        def dump(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(value):
                return
            save(name, value)

        module.register_forward_hook(dump)

    def register_mlp(layer, prefix):
        def dump_fc1(_module, _inputs, output):
            value, bias = output
            biased = value + bias if bias is not None else value
            save(f"{prefix}.mlp.c_fc", biased)
            save(f"{prefix}.mlp.gelu", layer.mlp.activation_func(biased))

        def dump_fc2(_module, _inputs, output):
            value, bias = output
            save(f"{prefix}.mlp.c_proj", value + bias if bias is not None else value)

        layer.mlp.linear_fc1.register_forward_hook(dump_fc1)
        layer.mlp.linear_fc2.register_forward_hook(dump_fc2)

    register(model.embedding.word_embeddings, "transformer.wte")
    register(model.embedding.position_embeddings, "transformer.wpe")
    register(model.embedding, "TransformerFirstStage")
    for i, layer in enumerate(model.decoder.layers):
        prefix = f"transformer.h.{i}"
        register(layer.input_layernorm, f"{prefix}.ln_1")
        register(layer.self_attention.linear_qkv, f"{prefix}.attn.c_attn")
        register(layer.self_attention.linear_proj, f"{prefix}.attn.c_proj")
        register(layer.pre_mlp_layernorm, f"{prefix}.ln_2")
        register_mlp(layer, prefix)
        register(layer, prefix)
    register(model.decoder.final_layernorm, "transformer.ln_f")
    register(model.output_layer, "lm_head")

def load_llmc(model, path):
    args = get_args()
    if args.tensor_model_parallel_size != 1 or args.pipeline_model_parallel_size != 1:
        raise RuntimeError("LLMC direct loading supports TP=1, PP=1")
    with open(path, "rb") as stream, torch.no_grad():
        header = struct.unpack("<256i", stream.read(1024))
        if header[:8] != (20240326, 3, 1024, 50257, 12, 12, 768, 50304):
            raise RuntimeError(f"unexpected LLMC header: {header[:8]}")
        padded_wte = read_tensor(stream, (50304, 768))
        assign(model.embedding.word_embeddings.weight, padded_wte[: model.vocab_size], "wte")
        assign(model.embedding.position_embeddings.weight, read_tensor(stream, (1024, 768)), "wpe")
        shapes = {
            "ln1w": (768,), "ln1b": (768,), "qkvw": (2304, 768), "qkvb": (2304,),
            "projw": (768, 768), "projb": (768,), "ln2w": (768,), "ln2b": (768,),
            "fc1w": (3072, 768), "fc1b": (3072,), "fc2w": (768, 3072), "fc2b": (768,),
        }
        values = {name: [read_tensor(stream, shape) for _ in range(12)] for name, shape in shapes.items()}
        final_w, final_b = read_tensor(stream, (768,)), read_tensor(stream, (768,))
        if stream.read(1):
            raise RuntimeError("trailing LLMC checkpoint data")
        for i, layer in enumerate(model.decoder.layers):
            if os.environ.get("LLMC_QKV_LAYOUT", "interleaved") == "block":
                qkvw, qkvb = values["qkvw"][i], values["qkvb"][i]
            else:
                qkvw = values["qkvw"][i].view(3, 12, 64, 768).permute(1, 0, 2, 3).reshape(2304, 768)
                qkvb = values["qkvb"][i].view(3, 12, 64).permute(1, 0, 2).reshape(2304)
            pairs = [
                (layer.input_layernorm.weight, values["ln1w"][i], "ln1w"),
                (layer.input_layernorm.bias, values["ln1b"][i], "ln1b"),
                (layer.self_attention.linear_qkv.weight, qkvw, "qkvw"),
                (layer.self_attention.linear_qkv.bias, qkvb, "qkvb"),
                (layer.self_attention.linear_proj.weight, values["projw"][i], "projw"),
                (layer.self_attention.linear_proj.bias, values["projb"][i], "projb"),
                (layer.pre_mlp_layernorm.weight, values["ln2w"][i], "ln2w"),
                (layer.pre_mlp_layernorm.bias, values["ln2b"][i], "ln2b"),
                (layer.mlp.linear_fc1.weight, values["fc1w"][i], "fc1w"),
                (layer.mlp.linear_fc1.bias, values["fc1b"][i], "fc1b"),
                (layer.mlp.linear_fc2.weight, values["fc2w"][i], "fc2w"),
                (layer.mlp.linear_fc2.bias, values["fc2b"][i], "fc2b"),
            ]
            for target, source, name in pairs:
                assign(target, source, f"layer{i}.{name}")
        assign(model.decoder.final_layernorm.weight, final_w, "lnf.weight")
        assign(model.decoder.final_layernorm.bias, final_b, "lnf.bias")
    print(f"loaded LLMC checkpoint: {path}", flush=True)

def llmc_model_provider(pre_process=True, post_process=True, vp_stage=None, config=None, pg_collection=None):
    model = model_provider(gpt_builder, pre_process, post_process, vp_stage, config, pg_collection)
    load_llmc(model, get_args().llmc_filepath)
    register_tensor_dumps(model)
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
