#!/usr/bin/env python3
"""Convert a Hugging Face Qwen3 checkpoint to InfiniTrain's LLMC format."""

import argparse
import json
import os
import struct

import torch
from safetensors.torch import load_file

K_QWEN3_MAGIC = 20240804
K_LLMC_FP32_VERSION = 4


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--tp-rank", type=int, default=0)
    return parser.parse_args()


def load_hf_weights(hf_path):
    index_path = os.path.join(hf_path, "model.safetensors.index.json")
    single_path = os.path.join(hf_path, "model.safetensors")
    if os.path.exists(index_path):
        with open(index_path) as stream:
            weight_map = json.load(stream)["weight_map"]
        state_dict = {}
        for filename in dict.fromkeys(weight_map.values()):
            shard = load_file(os.path.join(hf_path, filename), device="cpu")
            state_dict.update(shard)
            print(f"loaded {filename}: {len(shard)} tensors")
        return state_dict
    if os.path.exists(single_path):
        return load_file(single_path, device="cpu")
    return torch.load(os.path.join(hf_path, "pytorch_model.bin"), map_location="cpu", weights_only=True)


def float_slot(value):
    return struct.unpack("i", struct.pack("f", value))[0]


def write_header(stream, config):
    header = [0] * 256
    header[:16] = [
        K_QWEN3_MAGIC,
        K_LLMC_FP32_VERSION,
        config["max_position_embeddings"],
        config["vocab_size"],
        config["num_hidden_layers"],
        config["num_attention_heads"],
        config["num_key_value_heads"],
        config["hidden_size"],
        config["intermediate_size"],
        1,
        float_slot(config.get("rms_norm_eps", 1e-6)),
        float_slot(config.get("rope_theta", 1_000_000.0)),
        0,
        4,
        1,
        0,
    ]
    stream.write(struct.pack("256i", *header))


def write_tensor(stream, tensor, dimensions):
    value = tensor.detach().to(dtype=torch.float32, device="cpu").contiguous()
    if value.dim() != dimensions:
        raise ValueError(f"expected {dimensions}D tensor, got {tuple(value.shape)}")
    value.numpy().tofile(stream)


def require_keys(state_dict, config):
    keys = ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
    for layer in range(config["num_hidden_layers"]):
        prefix = f"model.layers.{layer}"
        keys.extend(
            [
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.self_attn.q_norm.weight",
                f"{prefix}.self_attn.k_norm.weight",
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
                f"{prefix}.self_attn.o_proj.weight",
                f"{prefix}.post_attention_layernorm.weight",
                f"{prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.up_proj.weight",
                f"{prefix}.mlp.down_proj.weight",
            ]
        )
    missing = sorted(set(keys) - state_dict.keys())
    if missing:
        raise KeyError(f"missing {len(missing)} Qwen3 tensors; first entries: {missing[:5]}")


def convert(hf_path, output_path):
    with open(os.path.join(hf_path, "config.json")) as stream:
        config = json.load(stream)
    state_dict = load_hf_weights(hf_path)
    require_keys(state_dict, config)

    layers = config["num_hidden_layers"]
    with open(output_path, "wb") as stream:
        write_header(stream, config)
        write_tensor(stream, state_dict["model.embed_tokens.weight"], 2)

        for layer in range(layers):
            write_tensor(stream, state_dict[f"model.layers.{layer}.input_layernorm.weight"], 1)

        for layer in range(layers):
            prefix = f"model.layers.{layer}.self_attn"
            write_tensor(stream, state_dict[f"{prefix}.q_norm.weight"], 1)
            write_tensor(stream, state_dict[f"{prefix}.k_norm.weight"], 1)

        for layer in range(layers):
            prefix = f"model.layers.{layer}.self_attn"
            qkv = torch.cat(
                [
                    state_dict[f"{prefix}.q_proj.weight"],
                    state_dict[f"{prefix}.k_proj.weight"],
                    state_dict[f"{prefix}.v_proj.weight"],
                ],
                dim=0,
            )
            write_tensor(stream, qkv, 2)

        for layer in range(layers):
            write_tensor(stream, state_dict[f"model.layers.{layer}.self_attn.o_proj.weight"], 2)

        for layer in range(layers):
            write_tensor(stream, state_dict[f"model.layers.{layer}.post_attention_layernorm.weight"], 1)

        for projection in ("gate_proj", "up_proj"):
            for layer in range(layers):
                write_tensor(stream, state_dict[f"model.layers.{layer}.mlp.{projection}.weight"], 2)

        for layer in range(layers):
            write_tensor(stream, state_dict[f"model.layers.{layer}.mlp.down_proj.weight"], 2)

        write_tensor(stream, state_dict["model.norm.weight"], 1)
        write_tensor(stream, state_dict["lm_head.weight"], 2)

    print(f"wrote {output_path}: {os.path.getsize(output_path)} bytes")


def main():
    args = parse_args()
    if args.tp_size != 1 or args.tp_rank != 0:
        raise ValueError("write one complete LLMC file with --tp-size=1 --tp-rank=0; loaders shard it at runtime")
    convert(args.hf_path, args.output)


if __name__ == "__main__":
    main()
