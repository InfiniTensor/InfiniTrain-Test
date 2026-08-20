#!/usr/bin/env python3
"""Convert an InfiniTrain/LLMC token file to a Megatron indexed dataset."""

import argparse
import pathlib
import struct
import sys

import numpy as np
import torch

try:
    import nvidia_resiliency_ext as nvrx
    if not hasattr(nvrx, "__version__"):
        nvrx.__version__ = "0.6.0"
except ModuleNotFoundError:
    pass


HEADER_BYTES = 1024
DTYPES = {20240520: (np.uint16, 2), 20240801: (np.int32, 4)}
SUPPORTED_VERSIONS = {1, 7}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-bin", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--megatron-path", required=True)
    return parser.parse_args()


def read_llmc_tokens(path):
    with open(path, "rb") as stream:
        header = stream.read(HEADER_BYTES)
        if len(header) != HEADER_BYTES:
            raise RuntimeError(f"short LLMC header: {path}")
        magic, version, num_tokens = struct.unpack_from("<iii", header)
        if magic not in DTYPES:
            raise RuntimeError(f"unsupported LLMC dataset magic: {magic}")
        dtype, item_size = DTYPES[magic]
        payload = stream.read()
    if version not in SUPPORTED_VERSIONS:
        raise RuntimeError(f"unsupported LLMC dataset version: {version}")
    expected = num_tokens * item_size
    if len(payload) != expected:
        raise RuntimeError(f"payload has {len(payload)} bytes, expected {expected}")
    return np.frombuffer(payload, dtype=dtype).astype(np.int32, copy=True)


def main():
    args = parse_args()
    megatron_path = pathlib.Path(args.megatron_path).resolve()
    sys.path.insert(0, str(megatron_path))
    from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder

    tokens = read_llmc_tokens(args.input_bin)
    output_prefix = pathlib.Path(args.output_prefix).resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    builder = IndexedDatasetBuilder(str(output_prefix) + ".bin", dtype=np.int32)

    # One document preserves the contiguous LLMC token stream. GPTDataset then
    # creates samples by taking seq_length + 1 adjacent tokens.
    builder.add_document(torch.from_numpy(tokens), [len(tokens)])
    builder.finalize(str(output_prefix) + ".idx")
    print(f"converted {len(tokens)} tokens to {output_prefix}.bin/.idx")


if __name__ == "__main__":
    main()
