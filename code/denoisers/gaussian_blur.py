from torchvision.transforms.functional import gaussian_blur
from .denoiser import Denoiser


class GaussianBlur(Denoiser):
  def __str__(self):
    return "Gaussian Blur"
  
  def denoise(self, x_hat, x=None, id=0):
    return gaussian_blur(x_hat, (5, 5))
