import math
import torch


class Schedule:
    def __init__(self, min: float = 0, max: float = 1):
        self.factor = max - min
        self.offset = min

    def __call__(self, t: float) -> float:
        raise NotImplementedError("Schedule must be implemented by subclass")
    
    def transform(self, x: float) -> float:
        return self.factor*x + self.offset


class Cos(Schedule):
    def __init__(self, min=0, max=1):
        super().__init__(min, max)

    def __str__(self):
        return "Cos"
    
    def __call__(self, t):
        x = math.cos((1 - t) * math.pi/2)**2
        return self.transform(x)


class Linear(Schedule):
    def __init__(self, min=0, max=1):
        super().__init__(min, max)

    def __str__(self):
        return "Linear"
    
    def __call__(self, t):
        return self.transform(t)
    

class DDPM(Schedule):
    def __init__(self, min=0, max=1, beta_start=1e-4, beta_end=1e-2, T=2000):
        super().__init__(min, max)

        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1 - betas
        self.alpha_bar = torch.cumprod(alphas, dim=0)
        self.T = T

    def __str__(self):
        return "DDPM"
    
    def __call__(self, t):
        # FIXME: round to avoid numerical issues
        t = round(t * self.T)
        alpha_bar = self.alpha_bar[self.T - t].item()
        return self.transform(alpha_bar)
