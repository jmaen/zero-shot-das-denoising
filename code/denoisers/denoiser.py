from typing import Dict, Any
import torch


class Denoiser:
    def name(self) -> str:
        return id(self)

    def denoise(self, x: torch.Tensor, y: torch.Tensor = None, **kwargs) -> torch.Tensor:
        raise NotImplementedError("Denoising function must be implemented by subclass")
