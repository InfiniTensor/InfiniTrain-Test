#!/usr/bin/env python3
"""Llama3 LLMC checkpoint reader and Megatron tensor-layout helpers."""

from dataclasses import dataclass
import os
import struct

import numpy as np

MAGIC = 20240803
FP32_VERSION = 3
HEADER_BYTES = 1024


@dataclass(frozen=True)
class Llama3Config:
    max_position_embeddings: int
    vocab_size: int
    num_layers: int
    num_attention_heads: int
    num_query_groups: int
    hidden_size: int
    ffn_multiplier: float
    multiple_of: int
    norm_epsilon: float
    rotary_base: float
    use_scaled_rope: bool

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads

    @property
    def qkv_rows(self):
        return self.hidden_size + 2 * self.num_query_groups * self.head_dim

    @property
    def intermediate_size(self):
        hidden = 2 * (4 * self.hidden_size) // 3
        hidden = round(self.ffn_multiplier * hidden)
        return (hidden + self.multiple_of - 1) // self.multiple_of * self.multiple_of


class Reader:
    def __init__(self, path):
        self.path = os.fspath(path)
        self.data = np.memmap(self.path, mode="r", dtype=np.uint8)
        if self.data.size < HEADER_BYTES:
            raise ValueError(f"short Llama3 LLMC file: {self.path}")
        raw = struct.unpack("256i", self.data[:HEADER_BYTES])
        if raw[0] != MAGIC or raw[1] != FP32_VERSION:
            raise ValueError(f"unexpected Llama3 LLMC header: magic={raw[0]} version={raw[1]}")
        float_at = lambda index: struct.unpack("f", struct.pack("i", raw[index]))[0]
        self.config = Llama3Config(
            max_position_embeddings=raw[2], vocab_size=raw[3], num_layers=raw[4],
            num_attention_heads=raw[5], num_query_groups=raw[6], hidden_size=raw[7],
            ffn_multiplier=float_at(8), multiple_of=raw[9], norm_epsilon=float_at(10),
            rotary_base=float_at(11), use_scaled_rope=bool(raw[12]),
        )
        if self.config.hidden_size % self.config.num_attention_heads:
            raise ValueError("hidden size must be divisible by attention heads")
        self.offset = HEADER_BYTES
        self.tensors = self._index_tensors()
        if self.offset != self.data.size:
            raise ValueError(f"LLMC size mismatch: indexed {self.offset} bytes, file has {self.data.size}")

    def _take(self, name, shape):
        count = int(np.prod(shape))
        end = self.offset + count * 4
        if end > self.data.size:
            raise ValueError(f"short tensor {name}: shape={shape}")
        result = np.ndarray(shape, dtype=np.float32, buffer=self.data, offset=self.offset)
        self.offset = end
        return result

    def _index_tensors(self):
        cfg, layers, hidden = self.config, self.config.num_layers, self.config.hidden_size
        ffn = cfg.intermediate_size
        tensors = {"embedding": self._take("embedding", (cfg.vocab_size, hidden))}
        tensors["input_norm"] = [self._take(f"input_norm.{i}", (hidden,)) for i in range(layers)]
        tensors["qkv"] = [self._take(f"qkv.{i}", (cfg.qkv_rows, hidden)) for i in range(layers)]
        tensors["attention_output"] = [self._take(f"attention_output.{i}", (hidden, hidden)) for i in range(layers)]
        tensors["pre_mlp_norm"] = [self._take(f"pre_mlp_norm.{i}", (hidden,)) for i in range(layers)]
        tensors["up"] = [self._take(f"up.{i}", (ffn, hidden)) for i in range(layers)]
        tensors["gate"] = [self._take(f"gate.{i}", (ffn, hidden)) for i in range(layers)]
        tensors["down"] = [self._take(f"down.{i}", (hidden, ffn)) for i in range(layers)]
        tensors["final_norm"] = self._take("final_norm", (hidden,))
        tensors["output"] = self._take("output", (cfg.vocab_size, hidden))
        return tensors


def tensor_parallel_slice(value, rank, size, axis):
    if value.shape[axis] % size:
        raise ValueError(f"cannot shard shape {value.shape} axis {axis} over TP={size}")
    width = value.shape[axis] // size
    index = [slice(None)] * value.ndim
    index[axis] = slice(rank * width, (rank + 1) * width)
    return value[tuple(index)]


def pack_qkv_for_megatron(value, config, rank, size):
    groups = config.num_query_groups
    if groups % size:
        raise ValueError(f"query groups {groups} must be divisible by TP={size}")
    head_dim = config.head_dim
    queries_per_group = config.num_attention_heads // groups
    groups_per_rank = groups // size
    first, last = rank * groups_per_rank, (rank + 1) * groups_per_rank
    q_rows = config.hidden_size
    kv_rows = groups * head_dim
    q = value[:q_rows].reshape(groups, queries_per_group, head_dim, config.hidden_size)
    k = value[q_rows:q_rows + kv_rows].reshape(groups, 1, head_dim, config.hidden_size)
    v = value[q_rows + kv_rows:].reshape(groups, 1, head_dim, config.hidden_size)
    return np.concatenate([q[first:last], k[first:last], v[first:last]], axis=1).reshape(-1, config.hidden_size)
