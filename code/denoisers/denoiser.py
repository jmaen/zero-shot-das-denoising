import torch


class Denoiser:
  def denoise(self, x_hat: torch.Tensor, x: torch.Tensor = None, id: int = 0) -> torch.Tensor:
    raise NotImplementedError("Denoising function must be implemented by subclass")
