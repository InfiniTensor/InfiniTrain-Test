"""Run the Llama 3.2 1B PyTorch baseline on MUSA."""

import sys
from pathlib import Path

from musa_runtime import configure_musa, run_training_script


if __name__ == "__main__":
    configure_musa()
    run_training_script(Path(__file__).resolve().parents[1] / "train_llama3.2_1B.py", sys.argv[1:])
