"""MUSA operator adaptations for the tiny Mixtral baseline."""

import sys
from pathlib import Path

from megatron_musa import configure_megatron, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_megatron_mixtral as training


initialize = training.init_megatron


def init_megatron(args):
    initialize(args)
    from megatron.core.transformer.moe.moe_utils import RouterGatingLinearFunction

    # The local router explicitly chooses its computation dtype. Outer BF16
    # autocast would downcast its forward mm but leave its backward mm in FP32.
    forward = RouterGatingLinearFunction.forward
    RouterGatingLinearFunction.forward = staticmethod(
        torch.amp.autocast("musa", enabled=False)(forward)
    )

    # muDNN stable sort supports int32 keys. Converting boolean route masks
    # preserves their order, including ties.
    argsort = torch.Tensor.argsort

    def argsort_mask(tensor, *args, **kwargs):
        if tensor.device.type == "musa" and tensor.dtype == torch.bool:
            tensor = tensor.int()
        return argsort(tensor, *args, **kwargs)

    torch.Tensor.argsort = argsort_mask


if __name__ == "__main__":
    configure_megatron()
    training.init_megatron = init_megatron
    training.main()
