#!/usr/bin/env python3
"""Batch-compare InfiniTrain and Megatron training losses."""

import math
import re
import sys
from argparse import ArgumentParser
from pathlib import Path

from compare_utils import collect_flat_megatron_logs, collect_log_files, exit_if_duplicate_logs


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?|inf(?:inity)?|nan)"
INFINITRAIN_LOSS = re.compile(
    rf"step\s+(\d+)/(\d+)\s+\|\s+train loss\s+({NUMBER})", re.IGNORECASE
)


def get_dtype_from_filename(filename):
    if "_bfloat16" in filename or "_bf16" in filename:
        return "bfloat16"
    return "sp_fp32" if "_sp_" in filename else "fp32"


def parse_log(file_path):
    losses = {}
    expected_steps = None
    for line_number, line in enumerate(file_path.read_text(errors="replace").splitlines(), 1):
        match = INFINITRAIN_LOSS.search(line)
        if not match:
            continue
        step = int(match.group(1))
        declared_steps = int(match.group(2))
        loss = float(match.group(3))
        if step in losses:
            raise ValueError(f"{file_path}:{line_number}: duplicate step {step}")
        if expected_steps is not None and declared_steps != expected_steps:
            raise ValueError(
                f"{file_path}:{line_number}: inconsistent planned steps "
                f"{declared_steps}, expected {expected_steps}"
            )
        if not math.isfinite(loss):
            raise ValueError(f"{file_path}:{line_number}: non-finite loss at step {step}: {loss}")
        expected_steps = declared_steps
        losses[step] = loss
    if not losses:
        raise ValueError(f"no loss records found in {file_path}")
    return losses, expected_steps


def incomplete_steps_warning(label, values, expected_steps):
    if len(values) >= expected_steps:
        return None
    return (
        f"{label}: only {len(values)}/{expected_steps} steps found; "
        "comparing available steps"
    )


def compare_files(baseline_file, test_file, threshold):
    baseline, baseline_expected = parse_log(baseline_file)
    test, test_expected = parse_log(test_file)
    warnings = [
        warning
        for warning in (
            incomplete_steps_warning("baseline", baseline, baseline_expected),
            incomplete_steps_warning("test", test, test_expected),
        )
        if warning
    ]
    if baseline_expected != test_expected:
        warnings.append(
            f"planned step mismatch: baseline={baseline_expected}, test={test_expected}"
        )
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
    return len(all_steps), max_abs_diff, mismatches, warnings


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
    parser.add_argument("--include-cases", help="comma-separated log basenames to compare")
    parser.add_argument("--prefer-autocast-bfloat16", action="store_true", help="prefer historical Megatron autocast BF16 log names")
    parser.add_argument("--threshold-fp32", type=float, default=1e-5)
    parser.add_argument("--threshold-sp-fp32", type=float, default=1e-5)
    parser.add_argument("--threshold-bf16", type=float, default=1e-2)
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

    requested_missing = []
    if args.include_cases:
        selected = {name.strip() for name in args.include_cases.split(",") if name.strip()}
        selected = {name if name.endswith(".log") else f"{name}.log" for name in selected}
        requested_missing = sorted(selected - (set(baseline_files) | set(test_files)))
        baseline_files = {name: path for name, path in baseline_files.items() if name in selected}
        test_files = {name: path for name, path in test_files.items() if name in selected}

    baseline_only = sorted(set(baseline_files) - set(test_files))
    test_only = sorted(set(test_files) - set(baseline_files))
    common = sorted(set(baseline_files) & set(test_files))

    passed = []
    failed = []
    warnings = [f"requested case not found: {name}" for name in requested_missing]
    for name in common:
        dtype = get_dtype_from_filename(name)
        threshold = args.threshold_bf16 if dtype == "bfloat16" else (args.threshold_sp_fp32 if dtype == "sp_fp32" else args.threshold_fp32)
        try:
            steps, max_abs_diff, mismatches, case_warnings = compare_files(
                baseline_files[name], test_files[name], threshold
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
        warnings.extend(f"{name}: {message}" for message in case_warnings)

    print(f"Baseline: {args.baseline_dir.resolve()}")
    print(f"Test:     {test_display}")
    print()
    print_group(f"PASSED loss cases ({len(passed)})", passed)
    print_group(f"FAILED loss cases ({sum(row.startswith('[FAIL]') for row in failed)})", failed)
    if warnings:
        print_group(f"WARNINGS ({len(warnings)})", [f"WARNING: {message}" for message in warnings])

    if baseline_only:
        print(f"Missing from Megatron test: {', '.join(baseline_only)}")
    if test_only:
        print(f"Missing from InfiniTrain baseline: {', '.join(test_only)}")
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
    print(f"warnings: {len(warnings)}")
    print(f"total   : {total}")
    print("=" * 64)
    return 1 if failed_cases or baseline_only or test_only else 0


if __name__ == "__main__":
    sys.exit(main())
