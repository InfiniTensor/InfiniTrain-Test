#!/usr/bin/env python3
"""Qwen3 LLMC schema reader and Megatron layout helpers."""

from dataclasses import dataclass
import os
import struct

import numpy as np

MAGIC = 20240804
FP32_VERSIONS = (3, 4)
HEADER_BYTES = 1024


@dataclass(frozen=True)
class Qwen3Config:
    max_position_embeddings: int
    vocab_size: int
    num_layers: int
    num_attention_heads: int
    num_query_groups: int
    hidden_size: int
    intermediate_size: int | None
    norm_epsilon: float
    rotary_base: float

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads

    @property
    def qkv_rows(self):
        return self.hidden_size + 2 * self.num_query_groups * self.head_dim


class Reader:
    def __init__(self, path, ffn_hidden_size):
        self.path = os.fspath(path)
        self.data = np.memmap(self.path, mode="r", dtype=np.uint8)
        if self.data.size < HEADER_BYTES:
            raise ValueError(f"short Qwen3 LLMC file: {self.path}")
        raw = struct.unpack("256i", self.data[:HEADER_BYTES])
        if raw[0] != MAGIC or raw[1] not in FP32_VERSIONS:
            raise ValueError(f"unexpected Qwen3 LLMC header: magic={raw[0]} version={raw[1]}")
        float_at = lambda index: struct.unpack("f", struct.pack("i", raw[index]))[0]
        self.config = Qwen3Config(
            max_position_embeddings=raw[2],
            vocab_size=raw[3],
            num_layers=raw[4],
            num_attention_heads=raw[5],
            num_query_groups=raw[6],
            hidden_size=raw[7],
            intermediate_size=raw[8] if raw[1] >= 4 else None,
            norm_epsilon=float_at(10),
            rotary_base=float_at(11),
        )
        if self.config.hidden_size % self.config.num_attention_heads:
            raise ValueError("hidden size must be divisible by attention heads")
        self.ffn_hidden_size = ffn_hidden_size
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
        cfg = self.config
        hidden = cfg.hidden_size
        head_dim = cfg.head_dim
        layers = cfg.num_layers
        tensors = {"embedding": self._take("embedding", (cfg.vocab_size, hidden))}
        tensors["input_norm"] = [self._take(f"input_norm.{i}", (hidden,)) for i in range(layers)]
        tensors["q_norm"] = []
        tensors["k_norm"] = []
        for i in range(layers):
            tensors["q_norm"].append(self._take(f"q_norm.{i}", (head_dim,)))
            tensors["k_norm"].append(self._take(f"k_norm.{i}", (head_dim,)))
        tensors["qkv"] = [self._take(f"qkv.{i}", (cfg.qkv_rows, hidden)) for i in range(layers)]
        tensors["attention_output"] = [
            self._take(f"attention_output.{i}", (hidden, hidden)) for i in range(layers)
        ]
        tensors["pre_mlp_norm"] = [self._take(f"pre_mlp_norm.{i}", (hidden,)) for i in range(layers)]
        tensors["gate"] = [
            self._take(f"gate.{i}", (self.ffn_hidden_size, hidden)) for i in range(layers)
        ]
        tensors["up"] = [self._take(f"up.{i}", (self.ffn_hidden_size, hidden)) for i in range(layers)]
        tensors["down"] = [
            self._take(f"down.{i}", (hidden, self.ffn_hidden_size)) for i in range(layers)
        ]
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
    first_group = rank * groups_per_rank
    last_group = first_group + groups_per_rank

    q_rows = config.hidden_size
    kv_rows = groups * head_dim
    q = value[:q_rows].reshape(groups, queries_per_group, head_dim, config.hidden_size)
    k = value[q_rows : q_rows + kv_rows].reshape(groups, 1, head_dim, config.hidden_size)
    v = value[q_rows + kv_rows :].reshape(groups, 1, head_dim, config.hidden_size)
    packed = np.concatenate([q[first_group:last_group], k[first_group:last_group], v[first_group:last_group]], axis=1)
    return packed.reshape(-1, config.hidden_size)
