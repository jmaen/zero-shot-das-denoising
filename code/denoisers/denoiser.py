from typing import Dict, Any
import torch


class Denoiser:
  def denoise(self, x_hat: torch.Tensor, x: torch.Tensor = None, options: Dict[str, Any] = {}) -> torch.Tensor:
    raise NotImplementedError("Denoising function must be implemented by subclass")
