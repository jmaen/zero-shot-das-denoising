import time
import math
from typing import Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity
from utils import Logger
from .denoiser import Denoiser


class BaseDIP(Denoiser):
    def __init__(
        self,
        net: nn.Module,
        input_size: int = 3,
        lr: float = 0.01,
    ):
        self.net = net
        self.input_size = input_size
        self.lr = lr

        self.device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.mse = nn.MSELoss()

        self.logger = Logger()
        self.metrics = {
            "psnr": PeakSignalNoiseRatio().to(self.device),
            "ssim": StructuralSimilarityIndexMeasure().to(self.device),
            "lpips": LearnedPerceptualImagePatchSimilarity().to(self.device),
        }

    def denoise(self, x_hat, x=None, options={}):
        self.net.to(self.device)

        optimizer = optim.Adam(self.net.parameters(), self.lr)

        x_hat = x_hat.to(self.device)
        if x is not None:
            x = x.to(self.device)

        state = {
            "x_hat": x_hat,
            "x": x, 
            "x_out": None,
            "start": time.time(),
            "epoch": 0, 
            "metrics": {},
            "summary": {},
            "options": options,
        }

        z = self.init_z(state)

        self.net.train()

        self.on_train_start(state)
        while not self.should_stop(state):
            optimizer.zero_grad()

            x_out = self.net(z)
            loss = self.calculate_loss(x_out, x_hat)
            loss.backward()
            optimizer.step()

            state["epoch"] += 1
            state["x_out"] = x_out.detach()
            state["metrics"]["loss"] = loss.item()
            self.on_epoch_end(state)

            z = self.update_z(z, state)

        self.on_train_end(state)

        return state["x_out"]
    
    def init_z(self, state: Dict[str, Any]) -> torch.Tensor:
        return torch.rand_like(state["x_hat"], device=self.device) * 0.1
    
    def update_z(self, z: torch.Tensor, state: Dict[str, Any]) -> torch.Tensor:
        return z
    
    def calculate_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.mse(x, y)
    
    def should_stop(self, state: Dict[str, Any]) -> bool:
        raise NotImplementedError("Stopping function must be implemented by subclass")

    def on_train_start(self, state: Dict[str, Any]):
        state["options"]["config"].update({
            "variant": str(self),
            "architecture": str(self.net),
        })

        self.logger.init_run(state["options"]["mode"], state["options"]["config"])

    def on_epoch_end(self, state: Dict[str, Any]):
        for key in state["options"]["metrics"]:
            metric = self.metrics[key](state["x_out"], state["x"]).item()
            state["metrics"][key] = metric

        self.logger.log(state["metrics"])
    
    def on_train_end(self, state: Dict[str, Any]):
        duration = time.time() - state["start"]
        runtime = time.strftime('%H:%M:%S', time.gmtime(duration))
        state["summary"]["runtime"] = runtime

        for key in state["options"]["metrics"]:
            metric = self.metrics[key](state["x_out"], state["x"]).item()
            state["summary"][key] = metric

        self.logger.finish(state["summary"])


class DIP(BaseDIP):
    def __init__(self, net, input_size=3, lr=0.01, max_epochs=2400):
        super().__init__(net, input_size, lr)

        self.max_epochs = max_epochs
        self.mse = nn.MSELoss()

    def __str__(self):
        return f"DIP ({self.max_epochs})"
    
    def should_stop(self, state):
        return state["epoch"] >= self.max_epochs
    

class DIP_MWV(BaseDIP):
    def __init__(self, net, input_size=3, lr=0.01, window_size=100, patience=1000):
        super().__init__(net, input_size, lr)
         
        self.window_size = window_size
        self.patience = patience
        self.mse = nn.MSELoss()

    def __str__(self):
        return f"DIP (MWV)"
    
    def should_stop(self, state):
        if state["epoch"] >= state["epoch_opt"] + self.patience:
            state["x_out"] = state["x_opt"]
            return True
    
    def on_train_start(self, state):
        state["queue"] = []
        state["var_opt"] = torch.inf
        state["epoch_opt"] = 0
        state["x_opt"] = None

        super().on_train_start(state)

    def on_epoch_end(self, state):
        queue = state["queue"]
        queue.append(state["x_out"])
        if len(queue) > self.window_size:
            queue.pop(0)

            queue = torch.stack(queue)
            var = torch.sum(queue.var(dim=0)).item()

            state["metrics"]["var"] = var

            if var < state["var_opt"]:
                state["var_opt"] = var
                state["epoch_opt"] = state["epoch"]
                state["x_opt"] = state["x_out"]

        super().on_epoch_end(state)

    def on_train_end(self, state):
        state["summary"]["stopping_point"] = state["epoch_opt"]

        super().on_train_end(state)

class DIP_TV(BaseDIP):
    def __init__(self, net, input_size=3, lr=0.01, max_epochs=2400, alpha=1):
        super().__init__(net, input_size, lr)

        self.max_epochs = max_epochs
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def __str__(self):
        return f"DIP-TV ({self.max_epochs})"

    def calculate_loss(self, x, y):
        return super().calculate_loss(x, y) + self.alpha*self._tv_norm(x)
    
    def should_stop(self, state):
        return state["epoch"] >= self.max_epochs

    def _tv_norm(self, x):
        diff_v = torch.abs(x[:, :-1, :] - x[:, 1:, :])
        diff_h = torch.abs(x[:, :, :-1] - x[:, :, 1:])

        tv_norm = torch.sum(diff_v) + torch.sum(diff_h)

        normalized = tv_norm / x.sum()

        return normalized.item()


class DDIP(BaseDIP):
    # FIXME sometimes quality drops significantly for t ~ T

    def __init__(self, net, input_size=3, lr=0.01, T=2400):
        super().__init__(net, input_size, lr)

        self.T = T
        self.T_ = T - 20

    def __str__(self):
        return f"DDIP ({self.T})"
    
    def should_stop(self, state):
        return state["epoch"] >= self.T_
    
    def init_z(self, state):
        x = state["x_hat"]
        y = torch.randn_like(state["x_hat"], device=self.device)
        return self._cos_schedule(x, y, 0)

    def update_z(self, z, state):
        x = state["x_out"]
        y = torch.randn_like(state["x_out"], device=self.device)
        return self._cos_schedule(x, y, state["epoch"])

    def _cos_schedule(self, x, y, t):
        alpha_bar = math.cos((math.pi * (self.T_ - t)) / (2 * (self.T)))**2
        return math.sqrt(alpha_bar)*x + math.sqrt(1 - alpha_bar)*y


class SelfDIP(BaseDIP):
    def __init__(self, net, input_size=3, lr=0.01, T=2400):
        super().__init__(net, input_size, lr)

        self.T = T

    def __str__(self):
        return f"RDIP ({self.T})"
    
    def should_stop(self, state):
        return state["epoch"] >= self.T

    def update_z(self, z, state):
        return state["x_out"]

