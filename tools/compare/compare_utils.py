from pathlib import Path
import sys


def collect_log_files(base_dir: Path, include_prefix=None, prefer_autocast_bfloat16=False):
    """Collect comparable training logs keyed by basename."""
    files = {}
    duplicates = {}

    for path in base_dir.rglob("*.log"):
        if include_prefix and not path.name.startswith(include_prefix):
            continue
        if path.name == "run_metadata.log":
            continue
        if path.name.startswith(("build", "ctest_")) or path.name.endswith("_profile.log"):
            continue

        key = path.name
        is_autocast = key.endswith("_autocast_bfloat16.log")
        if prefer_autocast_bfloat16 and is_autocast:
            key = key.removesuffix("_autocast_bfloat16.log") + "_bfloat16.log"
        elif prefer_autocast_bfloat16 and key.endswith("_bfloat16.log"):
            autocast = path.with_name(key.removesuffix("_bfloat16.log") + "_autocast_bfloat16.log")
            if autocast.is_file():
                continue
        if key in files:
            duplicates.setdefault(key, [files[key]]).append(path)
            continue
        files[key] = path

    return files, duplicates


def collect_flat_megatron_logs(log_dir: Path):
    """Pair InfiniTrain and Megatron logs from the repository's flat log layout."""
    baseline_files = {}
    test_files = {}
    duplicates = {}

    for path in log_dir.rglob("infinitrain_*.log"):
        key = path.name.removeprefix("infinitrain_")
        if key in baseline_files:
            duplicates.setdefault(key, [baseline_files[key]]).append(path)
            continue
        baseline_files[key] = path

    for key in baseline_files:
        canonical = log_dir / key
        if key.endswith("_bfloat16.log"):
            autocast_name = key.removesuffix("_bfloat16.log") + "_autocast_bfloat16.log"
            autocast = log_dir / autocast_name
            test_files[key] = autocast if autocast.is_file() else canonical
        else:
            test_files[key] = canonical

    test_files = {key: path for key, path in test_files.items() if path.is_file()}
    return baseline_files, test_files, duplicates


def exit_if_duplicate_logs(base_dir: Path, duplicates):
    """Abort when duplicate basenames make comparison ambiguous."""
    if not duplicates:
        return

    print(f"Found duplicate log basenames in {base_dir.resolve()}, cannot compare safely:")
    for name, paths in sorted(duplicates.items()):
        print(f"  {name}: {', '.join(str(p.relative_to(base_dir)) for p in paths)}")
    sys.exit(1)
