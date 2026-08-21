#!/usr/bin/env python3
"""Run upstream Megatron GPT training from an InfiniTrain Qwen3 LLMC checkpoint."""

import os
import pathlib
import sys

import numpy as np
import torch

if os.environ.get("QWEN3_DISABLE_TF32") == "1":
    torch.backends.cuda.matmul.fp32_precision = "ieee"

try:
    import nvidia_resiliency_ext as nvrx
    if not hasattr(nvrx, "__version__"):
        nvrx.__version__ = "0.6.0"
except ModuleNotFoundError:
    pass

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[4]
ADAPTER_DIR = SCRIPT_DIR.parent / "adapter"
MEGATRON_PATH = pathlib.Path(os.environ.get("MEGATRON_PATH", REPO_ROOT / "third_party" / "Megatron-LM"))
sys.path.insert(0, str(MEGATRON_PATH))
sys.path.insert(0, str(ADAPTER_DIR))

import pretrain_gpt as upstream
from gpt_builders import gpt_builder
from megatron.core import parallel_state
from megatron.core.datasets import gpt_dataset
from megatron.core.enums import ModelType
from megatron.training import get_args, pretrain
from megatron.training.argument_utils import pretrain_cfg_container_from_args
from megatron.training.arguments import parse_and_validate_args
from model_provider import model_provider

from llmc_loader import Reader, pack_qkv_for_megatron, tensor_parallel_slice


def add_args(parser):
    parser.add_argument("--llmc-filepath", required=True)
    return parser


def ordered_shuffle_index(num_samples, total_size, numpy_random_state):
    del num_samples, numpy_random_state
    return np.arange(total_size, dtype=np.uint32)


def assign(target, source, name):
    source = np.asarray(source)
    if tuple(target.shape) != tuple(source.shape):
        raise RuntimeError(f"{name}: Megatron {tuple(target.shape)} != LLMC {tuple(source.shape)}")
    target.copy_(torch.from_numpy(source.copy()).to(device=target.device, dtype=target.dtype))


def register_tensor_dumps(model):
    output_dir = os.environ.get("QWEN3_TENSOR_DUMP_DIR")
    if not output_dir or parallel_state.get_tensor_model_parallel_rank() != 0:
        return
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    counters = {}

    def register(module, name):
        def dump(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(value):
                return
            index = counters.get(name, 0)
            counters[name] = index + 1
            suffix = f"_{index}" if index else ""
            np.save(output_dir / f"{name}{suffix}_forward.npy", value.detach().float().cpu().numpy())

        module.register_forward_hook(dump)

    register(model.embedding.word_embeddings, "transformer.wte")
    for i, layer in enumerate(model.decoder.layers):
        prefix = f"transformer.h.{i}"
        register(layer.input_layernorm, f"{prefix}.ln_1")
        register(layer.self_attention.linear_qkv, f"{prefix}.attn.c_attn")
        register(layer.self_attention.q_layernorm, f"{prefix}.attn.q_norm")
        register(layer.self_attention.k_layernorm, f"{prefix}.attn.k_norm")
        register(layer.self_attention.linear_proj, f"{prefix}.attn.c_proj")
        register(layer.pre_mlp_layernorm, f"{prefix}.ln_2")
        register(layer.mlp.linear_fc1, f"{prefix}.mlp.fc1_packed")
        register(layer.mlp.linear_fc2, f"{prefix}.mlp.c_proj")
    register(model.decoder.final_layernorm, "transformer.ln_f")
    register(model.output_layer, "lm_head")


def load_llmc(model, path):
    args = get_args()
    if args.pipeline_model_parallel_size != 1:
        raise RuntimeError("Qwen3 LLMC runtime loading currently supports PP=1")
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    reader = Reader(path, args.ffn_hidden_size)
    cfg = reader.config
    expected = (
        args.num_layers,
        args.hidden_size,
        args.num_attention_heads,
        args.num_query_groups,
        args.padded_vocab_size,
    )
    actual = (cfg.num_layers, cfg.hidden_size, cfg.num_attention_heads, cfg.num_query_groups, cfg.vocab_size)
    if actual != expected:
        raise RuntimeError(f"Qwen3 config mismatch: LLMC={actual}, Megatron={expected}")
    if cfg.intermediate_size is not None and cfg.intermediate_size != args.ffn_hidden_size:
        raise RuntimeError(
            f"Qwen3 FFN mismatch: LLMC={cfg.intermediate_size}, Megatron={args.ffn_hidden_size}"
        )

    tensors = reader.tensors
    with torch.no_grad():
        assign(model.embedding.word_embeddings.weight, tensor_parallel_slice(tensors["embedding"], tp_rank, tp_size, 0), "embedding")
        for i, layer in enumerate(model.decoder.layers):
            assign(layer.input_layernorm.weight, tensors["input_norm"][i], f"layer{i}.input_norm")
            assign(layer.self_attention.q_layernorm.weight, tensors["q_norm"][i], f"layer{i}.q_norm")
            assign(layer.self_attention.k_layernorm.weight, tensors["k_norm"][i], f"layer{i}.k_norm")
            assign(
                layer.self_attention.linear_qkv.weight,
                pack_qkv_for_megatron(tensors["qkv"][i], cfg, tp_rank, tp_size),
                f"layer{i}.qkv",
            )
            assign(
                layer.self_attention.linear_proj.weight,
                tensor_parallel_slice(tensors["attention_output"][i], tp_rank, tp_size, 1),
                f"layer{i}.attention_output",
            )
            assign(layer.pre_mlp_layernorm.weight, tensors["pre_mlp_norm"][i], f"layer{i}.pre_mlp_norm")
            gate = tensor_parallel_slice(tensors["gate"][i], tp_rank, tp_size, 0)
            up = tensor_parallel_slice(tensors["up"][i], tp_rank, tp_size, 0)
            assign(layer.mlp.linear_fc1.weight, np.concatenate([gate, up], axis=0), f"layer{i}.linear_fc1")
            assign(
                layer.mlp.linear_fc2.weight,
                tensor_parallel_slice(tensors["down"][i], tp_rank, tp_size, 1),
                f"layer{i}.linear_fc2",
            )
        assign(model.decoder.final_layernorm.weight, tensors["final_norm"], "final_norm")
        assign(model.output_layer.weight, tensor_parallel_slice(tensors["output"], tp_rank, tp_size, 0), "output")
    print(f"loaded Qwen3 LLMC checkpoint on TP rank {tp_rank}/{tp_size}: {path}", flush=True)


def llmc_model_provider(pre_process=True, post_process=True, vp_stage=None, config=None, pg_collection=None):
    model = model_provider(gpt_builder, pre_process, post_process, vp_stage, config, pg_collection)
    load_llmc(model, get_args().llmc_filepath)
    register_tensor_dumps(model)
    return model


if __name__ == "__main__":
    gpt_dataset._build_shuffle_index = ordered_shuffle_index
    setattr(upstream.train_valid_test_datasets_provider, "is_distributed", True)
    args = parse_and_validate_args(extra_args_provider=add_args, args_defaults={"tokenizer_type": "NullTokenizer"})
    pretrain(
        pretrain_cfg_container_from_args(args),
        upstream.train_valid_test_datasets_provider,
        llmc_model_provider,
        ModelType.encoder_or_decoder,
        upstream.forward_step,
        get_embedding_ranks=upstream.get_embedding_ranks,
    )
