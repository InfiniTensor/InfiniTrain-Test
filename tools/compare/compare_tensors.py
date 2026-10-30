#!/usr/bin/env python3
"""Compare matching InfiniTrain and Megatron forward tensor dumps."""

import argparse
import pathlib
import re

import numpy as np


def align_layout(reference, candidate):
    if reference.shape == candidate.shape:
        return candidate
    if reference.ndim >= 2:
        axes = list(range(candidate.ndim))
        axes[0], axes[1] = axes[1], axes[0]
        transposed = candidate.transpose(axes)
        if reference.shape == transposed.shape:
            return transposed
    if reference.ndim + 1 == candidate.ndim and reference.shape == candidate.shape[1:]:
        if np.array_equal(candidate, np.broadcast_to(candidate[0], candidate.shape)):
            return candidate[0]
    raise ValueError(f"shape mismatch: {reference.shape} != {candidate.shape}")


def execution_order(name):
    if name == "TransformerFirstStage_forward.npy":
        return (-1, 0)
    match = re.match(r"transformer\.h\.(\d+)\.(.+)_forward\.npy", name)
    if match:
        stage = {
            "ln_1": 0,
            "ln_2": 1,
            "mlp.c_fc": 2,
            "mlp.gelu": 3,
            "mlp.c_proj": 4,
        }.get(match.group(2), 99)
        return (int(match.group(1)), stage)
    layer = re.match(r"transformer\.h\.(\d+)_forward\.npy", name)
    if layer:
        return (int(layer.group(1)), 5)
    return (100, name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infinitrain-dir", type=pathlib.Path, required=True)
    parser.add_argument("--megatron-dir", type=pathlib.Path, required=True)
    parser.add_argument("--threshold", type=float, default=1e-5)
    args = parser.parse_args()

    names = sorted(
        (
            path.name
            for path in args.megatron_dir.glob("*_forward.npy")
            if ".attn." not in path.name
        ),
        key=execution_order,
    )
    if not names:
        raise ValueError(f"no Megatron tensor dumps in {args.megatron_dir}")

    first_failure = None
    print(f"{'tensor':58} {'max_abs':>12} {'mean_abs':>12} {'rms':>12}")
    for name in names:
        reference_path = args.infinitrain_dir / name
        if not reference_path.exists():
            continue
        reference = np.load(reference_path).astype(np.float32, copy=False)
        candidate = align_layout(reference, np.load(args.megatron_dir / name).astype(np.float32, copy=False))
        delta = np.abs(reference - candidate)
        max_abs = float(delta.max())
        mean_abs = float(delta.mean())
        rms = float(np.sqrt(np.mean(np.square(delta, dtype=np.float64))))
        print(f"{name:58} {max_abs:12.6g} {mean_abs:12.6g} {rms:12.6g}")
        if first_failure is None and max_abs > args.threshold:
            first_failure = (name, max_abs, mean_abs, rms)

    if first_failure:
        name, max_abs, mean_abs, rms = first_failure
        print(f"first max-abs failure above {args.threshold:g}: {name} "
              f"max={max_abs:.6g} mean={mean_abs:.6g} rms={rms:.6g}")
        return 1
    print(f"all matching tensors pass max-abs threshold {args.threshold:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
