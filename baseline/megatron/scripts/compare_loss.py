#!/usr/bin/env python3
"""Batch-compare InfiniTrain and Megatron training losses."""

import re
import sys
from argparse import ArgumentParser
from pathlib import Path

from compare_utils import collect_flat_megatron_logs, collect_log_files, exit_if_duplicate_logs


INFINITRAIN_LOSS = re.compile(r"step\s+(\d+)/\d+\s+\|\s+train loss\s+([-+\d.eE]+)")
MEGATRON_LOSS = re.compile(r"iteration\s+(\d+)/\s*\d+.*?lm loss:\s*([-+\d.eE]+)")


def get_dtype_from_filename(filename):
    return "bfloat16" if "_bfloat16" in filename else "fp32"


def parse_log(file_path):
    losses = {}
    for line_number, line in enumerate(file_path.read_text(errors="replace").splitlines(), 1):
        match = INFINITRAIN_LOSS.search(line) or MEGATRON_LOSS.search(line)
        if not match:
            continue
        step = int(match.group(1))
        if step in losses:
            raise ValueError(f"{file_path}:{line_number}: duplicate step {step}")
        losses[step] = float(match.group(2))
    if not losses:
        raise ValueError(f"no loss records found in {file_path}")
    return losses


def compare_files(baseline_file, test_file, threshold, correct_megatron_step2):
    baseline = parse_log(baseline_file)
    test = parse_log(test_file)
    if correct_megatron_step2:
        if 1 not in test or 2 not in test:
            raise ValueError(f"{test_file}: step-2 correction requires steps 1 and 2")
        test[2] = 2.0 * test[2] - test[1]

    all_steps = sorted(set(baseline) | set(test))
    mismatches = []
    max_abs_diff = 0.0
    for step in all_steps:
        if step not in baseline:
            mismatches.append(f"step {step} missing in baseline")
        elif step not in test:
            mismatches.append(f"step {step} missing in test")
        else:
            difference = abs(baseline[step] - test[step])
            max_abs_diff = max(max_abs_diff, difference)
            if difference > threshold:
                mismatches.append(
                    f"step {step}: {baseline[step]:.6f} vs {test[step]:.6f} "
                    f"(diff {difference:.2e})"
                )
    return len(all_steps), max_abs_diff, mismatches


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
    parser = ArgumentParser(description="Batch-compare training loss directories")
    parser.add_argument("baseline_dir", type=Path, help="InfiniTrain log directory")
    parser.add_argument("test_dir", type=Path, nargs="?", help="Megatron log directory; omit for flat-layout auto-pairing")
    parser.add_argument("--include-prefix", help="only compare log basenames with this prefix")
    parser.add_argument("--prefer-autocast-bfloat16", action="store_true", help="prefer historical Megatron autocast BF16 log names")
    parser.add_argument("--threshold-fp32", type=float, default=1e-5)
    parser.add_argument("--threshold-bf16", type=float, default=1e-2)
    parser.add_argument(
        "--no-megatron-step2-running-average",
        dest="correct_step2",
        action="store_false",
        help="disable Megatron iteration-2 running-average correction",
    )
    parser.set_defaults(correct_step2=True)
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

    baseline_only = sorted(set(baseline_files) - set(test_files))
    test_only = sorted(set(test_files) - set(baseline_files))
    common = sorted(set(baseline_files) & set(test_files))

    passed = []
    failed = []
    for name in common:
        dtype = get_dtype_from_filename(name)
        threshold = args.threshold_bf16 if dtype == "bfloat16" else args.threshold_fp32
        try:
            steps, max_abs_diff, mismatches = compare_files(
                baseline_files[name], test_files[name], threshold, args.correct_step2
            )
        except ValueError as error:
            failed.append(f"[FAIL] {name}: {error}")
            continue
        summary = (
            f"{name} ({dtype}) | max diff {max_abs_diff:.2e} "
            f"| threshold {threshold:.0e} | steps {steps}"
        )
        if mismatches:
            failed.append(f"[FAIL] {summary}")
            failed.extend(f"       {message}" for message in mismatches)
        else:
            passed.append(f"[PASS] {summary}")

    print(f"Baseline: {args.baseline_dir.resolve()}")
    print(f"Test:     {test_display}")
    print()
    print_group(f"PASSED loss cases ({len(passed)})", passed)
    print_group(f"FAILED loss cases ({sum(row.startswith('[FAIL]') for row in failed)})", failed)

    if baseline_only:
        print(f"Missing from test: {', '.join(baseline_only)}")
    if test_only:
        print(f"Missing from baseline: {', '.join(test_only)}")
    if baseline_only or test_only:
        print()

    total = len(common) + len(baseline_only) + len(test_only)
    failed_cases = sum(row.startswith("[FAIL]") for row in failed)
    passed_cases = len(passed)
    print("=" * 64)
    print(f"Summary: {passed_cases}/{total} cases passed")
    print(f"passed  : {passed_cases}")
    print(f"failed  : {failed_cases}")
    print(f"missing : {len(baseline_only) + len(test_only)}")
    print(f"total   : {total}")
    print("=" * 64)
    return 1 if failed_cases or baseline_only or test_only else 0


if __name__ == "__main__":
    sys.exit(main())
