"""MUSA runtime settings for the existing PyTorch training scripts."""

import os
import runpy
import sys
from contextlib import redirect_stdout
from pathlib import Path

# PyTorch can auto-load torch_musa. In either import path its logger should
# bind to stderr so runner logs contain training output.
with redirect_stdout(sys.stderr):
    import torch
    import torch_musa  # noqa: F401 - register the MUSA device and distributed backend
    import torchada  # noqa: F401 - activate the MUSA compatibility layer


def configure_musa():
    torch.cuda.is_available = torch.musa.is_available
    # Match the CUDA baseline's FP32 settings on both MUSA backends.
    torch.backends.musa.matmul.allow_tf32 = False
    torch.backends.mudnn.allow_tf32 = False
    primary_rank = os.getenv("RANK", "0") == "0"
    if primary_rank:
        print(
            f"PyTorch {torch.__version__} | {torch.musa.get_device_properties(0)} | MUSA matmul and muDNN TF32 disabled",
            file=sys.stderr,
            flush=True,
        )

    cross_entropy = torch.nn.functional.cross_entropy

    def baseline_cross_entropy(input, target, *args, **kwargs):
        # Match CUDA autocast for the baselines' flattened token logits and labels.
        if (input.dtype == torch.bfloat16 and input.ndim == 2 and target.ndim == 1
                and not args and kwargs.keys() <= {"ignore_index", "reduction"}):
            log_probs = torch.nn.functional.log_softmax(input, dim=1, dtype=input.dtype)
            return torch.nn.functional.nll_loss(log_probs.to(torch.float32), target, **kwargs)
        return cross_entropy(input, target, *args, **kwargs)

    torch.nn.functional.cross_entropy = baseline_cross_entropy
    if primary_rank:
        print("PyTorch baseline: BF16 log-softmax, FP32 NLL loss", file=sys.stderr, flush=True)

    # The legacy tensor-type API rejects PrivateUse1 tensor classes. The
    # PyTorch loader selects FP32 device tensors while constructing its model.
    set_tensor_type = torch.set_default_tensor_type

    def set_default_tensor_type(tensor_type):
        if tensor_type is torch.musa.FloatTensor:
            torch.set_default_dtype(torch.float32)
            torch.set_default_device("musa")
        else:
            torch.set_default_device(None)
            set_tensor_type(tensor_type)

    torch.set_default_tensor_type = set_default_tensor_type


def run_training_script(script, args):
    script = Path(script).resolve()
    sys.argv = [str(script), *args]
    sys.path.insert(0, str(script.parent))
    runpy.run_path(str(script), run_name="__main__")
