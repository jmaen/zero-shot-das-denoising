import math
import torch


class Schedule:
    def __call__(self, t: float) -> float:
        raise NotImplementedError("Schedule must be implemented by subclass")


class Cos(Schedule):
    def __init__(self, offset=0):
        self.offset = offset

    def __str__(self):
        return "Cos"
    
    def __call__(self, t):
        x = math.cos((1 - t) * math.pi/2)**2
        x = x*(1 - 2*self.offset) + self.offset
        return x


class Linear(Schedule):
    def __init__(self, offset=0):
        self.offset = offset

    def __str__(self):
        return "Linear"
    
    def __call__(self, t):
        return t*(1 - 2*self.offset) + self.offset
    

class DDPM(Schedule):
    def __init__(self, beta_start=1e-4, beta_end=1e-2, T=2000):
        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1 - betas
        self.alpha_bar = torch.cumprod(alphas, dim=0)
        self.T = T

    def __str__(self):
        return "DDPM"
    
    def __call__(self, t):
        # FIXME: round to avoid numerical issues
        t = round(t * self.T)
        return self.alpha_bar[self.T - t].item()
