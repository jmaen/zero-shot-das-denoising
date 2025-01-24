from torchvision.transforms.functional import gaussian_blur
from ..denoiser import Denoiser


class GaussianBlur(Denoiser):
    def name(self):
        return "Gaussian Blur"
  
    def denoise(self, y, **kwargs):
        return gaussian_blur(y, (5, 5))
