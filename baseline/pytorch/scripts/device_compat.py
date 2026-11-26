"""Accelerator compatibility helpers for the PyTorch baselines."""

import os
from contextlib import contextmanager, nullcontext

import torch

_ACCELERATOR = None
_TORCH_NPU_IMPORT_ERROR = None
_TRANSFER_TO_NPU = None


def _npu_available():
    global _TORCH_NPU_IMPORT_ERROR
    try:
        import torch_npu  # noqa: F401
    except (ImportError, OSError, RuntimeError) as error:
        _TORCH_NPU_IMPORT_ERROR = error
        return False
    return hasattr(torch, "npu") and torch.npu.is_available()


def _resolve_accelerator(requested):
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("INFINITRAIN_ACCELERATOR=cuda, but CUDA is not available")
        return "cuda"
    if requested == "npu":
        if not _npu_available():
            detail = f": {_TORCH_NPU_IMPORT_ERROR}" if _TORCH_NPU_IMPORT_ERROR else ""
            raise RuntimeError(
                f"INFINITRAIN_ACCELERATOR=npu, but torch_npu/NPU is not available{detail}"
            )
        return "npu"
    if requested == "cpu":
        return "cpu"
    if requested != "auto":
        raise RuntimeError(
            f"unsupported INFINITRAIN_ACCELERATOR={requested!r}; expected auto, cuda, npu or cpu"
        )

    if os.environ.get("ASCEND_HOME_PATH"):
        if _npu_available():
            return "npu"
        detail = f": {_TORCH_NPU_IMPORT_ERROR}" if _TORCH_NPU_IMPORT_ERROR else ""
        raise RuntimeError(
            "CANN is sourced, but torch_npu cannot see an available NPU"
            f"{detail}. Check the device mount and torch_npu compatibility."
        )
    if torch.cuda.is_available():
        return "cuda"
    if _npu_available():
        return "npu"
    return "cpu"


def initialize_accelerator():
    """Select the accelerator and install torch_npu's CUDA compatibility layer."""
    global _ACCELERATOR, _TRANSFER_TO_NPU
    if _ACCELERATOR is not None:
        return _ACCELERATOR

    requested = os.environ.get("INFINITRAIN_ACCELERATOR", "auto").lower()
    accelerator = _resolve_accelerator(requested)
    if accelerator == "npu":
        import torch_npu  # noqa: F401
        from torch_npu.contrib import transfer_to_npu

        _TRANSFER_TO_NPU = transfer_to_npu
    _ACCELERATOR = accelerator
    return accelerator


def accelerator_type():
    if _ACCELERATOR is None:
        return initialize_accelerator()
    return _ACCELERATOR


def is_available():
    return accelerator_type() != "cpu"


def distributed_backend():
    accelerator = accelerator_type()
    if accelerator == "npu":
        return "hccl"
    if accelerator == "cuda":
        return "nccl"
    return "gloo"


def device_name(rank=None):
    accelerator = accelerator_type()
    if accelerator == "cpu":
        return "cpu"
    if rank is None:
        rank = current_device()
    return f"{accelerator}:{rank}"


def set_device(rank):
    if accelerator_type() == "npu":
        torch.npu.set_device(rank)
    elif accelerator_type() == "cuda":
        torch.cuda.set_device(rank)


def current_device():
    if accelerator_type() == "npu":
        return torch.npu.current_device()
    if accelerator_type() == "cuda":
        return torch.cuda.current_device()
    return 0


def synchronize():
    if accelerator_type() == "npu":
        torch.npu.synchronize()
    elif accelerator_type() == "cuda":
        torch.cuda.synchronize()


def reset_peak_memory_stats():
    if accelerator_type() == "npu":
        torch.npu.reset_peak_memory_stats()
    elif accelerator_type() == "cuda":
        torch.cuda.reset_peak_memory_stats()


def max_memory_allocated_mib():
    if accelerator_type() == "npu":
        allocated = torch.npu.max_memory_allocated()
    elif accelerator_type() == "cuda":
        allocated = torch.cuda.max_memory_allocated()
    else:
        return 0
    return allocated // 1024 // 1024


def manual_seed(seed):
    torch.manual_seed(seed)
    if accelerator_type() == "npu":
        torch.npu.manual_seed_all(seed)
    elif accelerator_type() == "cuda":
        torch.cuda.manual_seed_all(seed)


def autocast_context(dtype):
    if dtype == torch.float32 or accelerator_type() == "cpu":
        return nullcontext()
    return torch.amp.autocast(device_type=accelerator_type(), dtype=dtype)


@contextmanager
def accelerated_default_device():
    """Create model parameters on the selected accelerator, then restore CPU defaults."""
    accelerator = accelerator_type()
    if accelerator == "cpu":
        yield
        return
    if accelerator == "npu":
        previous_device = torch.get_default_device()
        torch.set_default_device("npu")
        try:
            yield
        finally:
            torch.set_default_device(previous_device)
        return

    original_default_type = torch.get_default_dtype()
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    try:
        yield
    finally:
        torch.set_default_tensor_type(
            torch.tensor([], dtype=original_default_type, device="cpu").type()
        )


def configure_fp32():
    """Set the same IEEE-FP32 math policy on CUDA and NPU."""
    if accelerator_type() == "npu":
        matmul = torch.npu.matmul
        if hasattr(matmul, "allow_hf32"):
            matmul.allow_hf32 = False
        elif hasattr(matmul, "fp32_precision"):
            matmul.fp32_precision = "ieee"
    elif accelerator_type() == "cuda":
        matmul = torch.backends.cuda.matmul
        if hasattr(matmul, "allow_tf32"):
            matmul.allow_tf32 = False
        if hasattr(matmul, "allow_fp16_reduced_precision_reduction"):
            matmul.allow_fp16_reduced_precision_reduction = False
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch.backends.cudnn, "deterministic"):
            torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")
