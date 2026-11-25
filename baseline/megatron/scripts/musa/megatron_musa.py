#!/usr/bin/env python3
"""MUSA worker for the existing Megatron torchrun launchers."""

import json
import os
import sys
import typing
from pathlib import Path

from typing_extensions import override

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "pytorch/scripts/musa"))
from musa_runtime import configure_musa, run_training_script, torch


def redirect_amp_decorator(decorator):
    def decorate(*args, device_type, **kwargs):
        return decorator(
            *args,
            device_type="musa" if device_type == "cuda" else device_type,
            **kwargs,
        )

    return decorate


def configure_megatron():
    configure_musa()
    if sys.version_info < (3, 12):
        typing.override = override
    # torchada redirects torch.amp.autocast. Megatron also uses these AMP APIs.
    torch.autocast = torch.amp.autocast
    torch.amp.custom_fwd = redirect_amp_decorator(torch.amp.custom_fwd)
    torch.amp.custom_bwd = redirect_amp_decorator(torch.amp.custom_bwd)


if __name__ == "__main__":
    configure_megatron()
    # torchrun invokes PYTHON_EXEC with -u followed by the training script.
    script, *args = sys.argv[2:]
    worker_args = json.loads(os.getenv("MUSA_BASELINE_WORKER_ARGS", "[]"))
    run_training_script(script, [*args, *worker_args])
