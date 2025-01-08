from torchvision.transforms.functional import gaussian_blur
from .denoiser import Denoiser


class GaussianBlur(Denoiser):
  def key(self):
    return "Gaussian Blur"
  
  def denoise(self, x_hat, x=None, options={}):
    return gaussian_blur(x_hat, (5, 5))
