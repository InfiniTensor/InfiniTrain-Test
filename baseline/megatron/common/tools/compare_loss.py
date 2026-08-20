#!/usr/bin/env python3
"""Strictly compare InfiniTrain and Megatron training loss logs."""

import argparse
import json
import math
import pathlib
import re
import sys


INFINITRAIN = re.compile(r"step\s+(\d+)/\d+\s+\|\s+train loss\s+([-+\d.eE]+)")
MEGATRON = re.compile(r"iteration\s+(\d+)/\s*\d+.*?lm loss:\s*([-+\d.eE]+)")


def parse_losses(path):
    losses = {}
    for line_number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        match = INFINITRAIN.search(line) or MEGATRON.search(line)
        if not match:
            continue
        step, loss = int(match.group(1)), float(match.group(2))
        if step in losses:
            raise ValueError(f"{path}:{line_number}: duplicate step {step}")
        if not math.isfinite(loss):
            raise ValueError(f"{path}:{line_number}: non-finite loss {loss}")
        losses[step] = loss
    if not losses:
        raise ValueError(f"no loss records found in {path}")
    return losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infinitrain-log", type=pathlib.Path, required=True)
    parser.add_argument("--megatron-log", type=pathlib.Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--expected-steps", type=int)
    parser.add_argument("--output-json", type=pathlib.Path)
    parser.add_argument(
        "--megatron-step2-running-average",
        action="store_true",
        help="recover step 2 when Megatron logs the mean of steps 1 and 2",
    )
    args = parser.parse_args()

    lhs = parse_losses(args.infinitrain_log)
    rhs = parse_losses(args.megatron_log)
    if args.megatron_step2_running_average:
        if 1 not in rhs or 2 not in rhs:
            raise ValueError("Megatron step-2 correction requires steps 1 and 2")
        rhs[2] = 2.0 * rhs[2] - rhs[1]
    if lhs.keys() != rhs.keys():
        raise ValueError(f"step mismatch: InfiniTrain={sorted(lhs)}, Megatron={sorted(rhs)}")
    if args.expected_steps is not None and len(lhs) != args.expected_steps:
        raise ValueError(f"found {len(lhs)} steps, expected {args.expected_steps}")

    rows = []
    for step in sorted(lhs):
        diff = abs(lhs[step] - rhs[step])
        rows.append({"step": step, "infinitrain_loss": lhs[step], "megatron_loss": rhs[step], "abs_diff": diff})
    result = {
        "atol": args.atol,
        "steps": len(rows),
        "max_abs_diff": max(row["abs_diff"] for row in rows),
        "passed": all(row["abs_diff"] <= args.atol for row in rows),
        "megatron_step2_running_average_corrected": args.megatron_step2_running_average,
        "results": rows,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
