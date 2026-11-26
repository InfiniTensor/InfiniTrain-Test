"""Accelerator compatibility helpers for Megatron baseline entry points."""

import os
import sys

import torch

_ACCELERATOR = None
_TORCH_NPU_IMPORT_ERROR = None


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


def _load_npu_adaptor():
    for variable in ("MEGATRON_ADAPTOR_PATH", "MINDSPEED_PATH"):
        package_path = os.environ.get(variable)
        if package_path:
            sys.path.insert(0, os.path.abspath(package_path))
    try:
        import megatron_adaptor  # noqa: F401
    except Exception as error:
        raise RuntimeError(
            "Ascend Megatron baselines require MegatronAdaptor, TransformerEngineNPU "
            "and a compatible MindSpeed installation. Install the matching "
            "26.2.0_core_r0.18.0 components before importing Megatron."
        ) from error


def initialize_accelerator():
    """Select the accelerator and patch Megatron before its modules are imported."""
    global _ACCELERATOR
    if _ACCELERATOR is not None:
        return _ACCELERATOR

    requested = os.environ.get("INFINITRAIN_ACCELERATOR", "auto").lower()
    accelerator = _resolve_accelerator(requested)
    if accelerator == "npu":
        _load_npu_adaptor()
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
    return torch.device("cpu")


def synchronize():
    if accelerator_type() == "npu":
        torch.npu.synchronize()
    elif accelerator_type() == "cuda":
        torch.cuda.synchronize()


def empty_cache():
    if accelerator_type() == "npu":
        torch.npu.empty_cache()
    elif accelerator_type() == "cuda":
        torch.cuda.empty_cache()


def max_memory_allocated_mib():
    if accelerator_type() == "npu":
        allocated = torch.npu.max_memory_allocated()
    elif accelerator_type() == "cuda":
        allocated = torch.cuda.max_memory_allocated()
    else:
        return 0
    return allocated // 1024 // 1024


def autocast_device_type():
    accelerator = accelerator_type()
    return "cpu" if accelerator == "cpu" else accelerator


def manual_seed(seed):
    torch.manual_seed(seed)
    if accelerator_type() == "npu":
        torch.npu.manual_seed_all(seed)
    elif accelerator_type() == "cuda":
        torch.cuda.manual_seed_all(seed)


def timing_dtype():
    return torch.float32 if accelerator_type() == "npu" else torch.float64


def counter_dtype():
    return torch.int32 if accelerator_type() == "npu" else torch.int64


def disable_reduced_precision_fp32():
    """Match the CUDA IEEE-FP32 baseline policy with the available backend API."""
    if accelerator_type() == "npu":
        matmul = torch.npu.matmul
        if hasattr(matmul, "allow_hf32"):
            matmul.allow_hf32 = False
        elif hasattr(matmul, "fp32_precision"):
            matmul.fp32_precision = "ieee"
    elif accelerator_type() == "cuda":
        matmul = torch.backends.cuda.matmul
        if hasattr(matmul, "fp32_precision"):
            matmul.fp32_precision = "ieee"
        elif hasattr(matmul, "allow_tf32"):
            matmul.allow_tf32 = False
