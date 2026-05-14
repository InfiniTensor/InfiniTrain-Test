import torch
import numpy as np
from pathlib import Path


class PrecisionHook:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._handles = []

    def _save_tensor(self, name: str, stage: str, tensor: torch.Tensor):
        """Save tensor as numpy binary file"""
        data = tensor.detach().float().cpu().contiguous().numpy()
        filename = f"{name}_{stage}.npy"
        np.save(self.output_dir / filename, data)

    def _forward_hook(self, name: str):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                self._save_tensor(name, "forward", output)
        return hook

    def _backward_hook(self, name: str):
        def hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                self._save_tensor(name, "backward", grad_output[0])
        return hook

    def register(self, model: torch.nn.Module):
        for name, module in model.named_modules():
            # if isinstance(module, (torch.nn.Sequential, torch.nn.ModuleList, torch.nn.ModuleDict)):
                # continue
            if not name:
                continue
            h1 = module.register_forward_hook(self._forward_hook(name))
            h2 = module.register_full_backward_hook(self._backward_hook(name))
            self._handles.extend([h1, h2])

    def remove(self):
        for h in self._handles:
            h.remove()
