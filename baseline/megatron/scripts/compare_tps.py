#!/usr/bin/env python3
"""Batch-compare InfiniTrain and Megatron training throughput."""

import re
import sys
from argparse import ArgumentParser
from pathlib import Path

from compare_utils import collect_flat_megatron_logs, collect_log_files, exit_if_duplicate_logs


THROUGHPUT = re.compile(
    r"(?:PERF\s+)?step\s+(\d+)/\d+.*?\|\s+(\d+(?:\.\d+)?)\s+tok/s"
)


def parse_log(file_path):
    values = {}
    for line_number, line in enumerate(file_path.read_text(errors="replace").splitlines(), 1):
        match = THROUGHPUT.search(line)
        if not match:
            continue
        step = int(match.group(1))
        if step in values:
            raise ValueError(f"{file_path}:{line_number}: duplicate step {step}")
        values[step] = float(match.group(2))
    if not values:
        raise ValueError(f"no throughput records found in {file_path}")
    return values


def compare_files(baseline_file, test_file, warmup_steps):
    baseline = {step: value for step, value in parse_log(baseline_file).items() if step > warmup_steps}
    test = {step: value for step, value in parse_log(test_file).items() if step > warmup_steps}
    if not baseline or not test:
        raise ValueError("no throughput records remain after warmup")
    if baseline.keys() != test.keys():
        raise ValueError(
            f"step mismatch after warmup: baseline={sorted(baseline)}, test={sorted(test)}"
        )

    baseline_average = sum(baseline.values()) / len(baseline)
    test_average = sum(test.values()) / len(test)
    throughput_ratio = baseline_average / test_average
    return baseline_average, test_average, throughput_ratio, len(baseline)


def print_group(title, rows):
    print("=" * 64)
    print(title)
    print("=" * 64)
    if not rows:
        print("  (none)")
    for row in rows:
        print(row)
    print()


def main():
    parser = ArgumentParser(description="Batch-compare training throughput directories")
    parser.add_argument("baseline_dir", type=Path, help="InfiniTrain log directory")
    parser.add_argument("test_dir", type=Path, nargs="?", help="Megatron log directory; omit for flat-layout auto-pairing")
    parser.add_argument("--include-prefix", help="only compare log basenames with this prefix")
    parser.add_argument("--include-cases", help="comma-separated log basenames to compare")
    parser.add_argument("--prefer-autocast-bfloat16", action="store_true", help="prefer historical Megatron autocast BF16 log names")
    parser.add_argument("--warmup-steps", type=int, default=1)
    args = parser.parse_args()

    if args.test_dir is None:
        baseline_files, test_files, baseline_duplicates = collect_flat_megatron_logs(args.baseline_dir)
        test_duplicates = {}
        test_display = args.baseline_dir.resolve()
    else:
        baseline_files, baseline_duplicates = collect_log_files(args.baseline_dir, args.include_prefix)
        test_files, test_duplicates = collect_log_files(args.test_dir, args.include_prefix, args.prefer_autocast_bfloat16)
        test_display = args.test_dir.resolve()
    exit_if_duplicate_logs(args.baseline_dir, baseline_duplicates)
    if args.test_dir is not None:
        exit_if_duplicate_logs(args.test_dir, test_duplicates)

    if args.include_cases:
        selected = {name.strip() for name in args.include_cases.split(",") if name.strip()}
        selected = {name if name.endswith(".log") else f"{name}.log" for name in selected}
        baseline_files = {name: path for name, path in baseline_files.items() if name in selected}
        test_files = {name: path for name, path in test_files.items() if name in selected}

    baseline_only = sorted(set(baseline_files) - set(test_files))
    test_only = sorted(set(test_files) - set(baseline_files))
    common = sorted(set(baseline_files) & set(test_files))

    results = []
    errors = []
    for name in common:
        try:
            baseline_average, test_average, ratio, steps = compare_files(
                baseline_files[name], test_files[name], args.warmup_steps
            )
        except ValueError as error:
            errors.append(f"[ERROR] {name}: {error}")
            continue

        results.append(
            f"{name} | InfiniTrain {baseline_average:.2f} tok/s "
            f"| Megatron {test_average:.2f} tok/s | InfiniTrain/Megatron {ratio:.1%} "
            f"| steps {steps}"
        )

    print(f"Baseline: {args.baseline_dir.resolve()}")
    print(f"Test:     {test_display}")
    print(f"Warmup steps excluded: {args.warmup_steps}")
    print()
    print_group(f"THROUGHPUT RESULTS ({len(results)})", results)
    if errors:
        print_group(f"ERRORS ({len(errors)})", errors)

    if baseline_only:
        print(f"Missing from test: {', '.join(baseline_only)}")
    if test_only:
        print(f"Missing from baseline: {', '.join(test_only)}")
    if baseline_only or test_only:
        print()

    total = len(common) + len(baseline_only) + len(test_only)
    print("=" * 64)
    print(f"Summary: {len(results)}/{total} cases compared")
    print(f"compared: {len(results)}")
    print(f"errors  : {len(errors)}")
    print(f"missing : {len(baseline_only) + len(test_only)}")
    print(f"total   : {total}")
    print("=" * 64)
    return 1 if errors or baseline_only or test_only else 0


if __name__ == "__main__":
    sys.exit(main())
