import torch
from torch.nn.functional import mse_loss


class Loss:
    def __call__(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Loss function must be implemented by subclass")
    

class Composed(Loss):
    def __init__(self, loss1, loss2, alpha=1):
        self.loss1 = loss1
        self.loss2 = loss2
        self.alpha = alpha
    
    def __str__(self):
        return f"{str(self.loss1)} + {str(self.loss2)}"
    
    def __call__(self, x, y, z):
        return self.loss1(x, y, z) + self.alpha*self.loss2(x, y, z)


class MSE(Loss):
    def __str__(self):
        return "MSE"
    
    def __call__(self, x, y, z):
        return mse_loss(x, y)


class NMSE(Loss):
    def __str__(self):
        return "NMSE"
    
    def __call__(self, x, y, z):
        y = self._random_neighbors(y)
        return mse_loss(x, y)
    
    def _random_neighbors(self, y):
        _, _, H, W = y.shape
        
        offsets = [(-1, 0), (0, -1), (0, 1), (1, 0)]
        offsets = torch.tensor(offsets, device=y.device)
        random_offsets = offsets[torch.randint(0, len(offsets), (H, W), device=y.device)]
        
        v_indices = torch.arange(H, device=y.device).view(-1, 1).expand(H, W) + random_offsets[..., 0]
        h_indices = torch.arange(W, device=y.device).view(1, -1).expand(H, W) + random_offsets[..., 1]
        
        v_indices = torch.clamp(v_indices, 0, H - 1)
        h_indices = torch.clamp(h_indices, 0, W - 1)
        
        y = y[:, :, v_indices, h_indices]

        return y
    

class AE(Loss):
    def __str__(self):
        return "AE"
    
    def __call__(self, x, y, z):
        return mse_loss(x, z)


class TV(Loss):
    def __str__(self):
        return "TV"
    
    def __call__(self, x, y, z):
        return self._tv_norm(x)
    
    def _tv_norm(self, x):
        # FIXME
        diff_v = torch.abs(x[:, :-1, :] - x[:, 1:, :])
        diff_h = torch.abs(x[:, :, :-1] - x[:, :, 1:])

        tv_norm = torch.sum(diff_v) + torch.sum(diff_h)

        return tv_norm
