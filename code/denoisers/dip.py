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

    def key(self):
        return f"{str(self)} | {str(self.net)}"

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
            state["z"] = z.detach()

            self.optimize(z, x_hat, optimizer, state)

            z = self.update_z(z, state)

        self.on_train_end(state)

        self.net.reset_parameters()

        return state["x_out"]
    
    def optimize(self, input: torch.Tensor, target: torch.Tensor, optimizer: optim.Optimizer, state: Dict[str, Any]):
        optimizer.zero_grad()
        output = self.net(input)
        loss = self.calculate_loss(output, target, state)
        loss.backward()
        optimizer.step()

        state["epoch"] += 1
        state["x_out"] = output.detach()
        state["metrics"]["loss"] = loss.item()
        self.on_epoch_end(state)
    
    def init_z(self, state: Dict[str, Any]) -> torch.Tensor:
        return torch.rand_like(state["x_hat"], device=self.device) * 0.1
    
    def update_z(self, z: torch.Tensor, state: Dict[str, Any]) -> torch.Tensor:
        return z
    
    def calculate_loss(self, x: torch.Tensor, y: torch.Tensor, state: Dict[str, Any] = None) -> torch.Tensor:
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

        data = state["metrics"].copy()
        if state["options"]["save_images"]:
            data.update({"input & output": [state["z"], state["x_out"]]})

        self.logger.log(data)
    
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
    def __init__(self, net, input_size=3, lr=0.01, window_size=100, patience=1000, z_init="gaussian"):
        super().__init__(net, input_size, lr)
         
        self.window_size = window_size
        self.patience = patience
        self.z_init = z_init
        self.mse = nn.MSELoss()

    def __str__(self):
        return f"DIP MVW ({self.z_init})"
    
    def init_z(self, state: Dict[str, Any]) -> torch.Tensor:
        if self.z_init == "uniform":
            z = torch.rand_like(state["x_hat"], device=self.device) * 0.1
        elif self.z_init == "gaussian":
            z = torch.randn_like(state["x_hat"], device=self.device)

        return z
    
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


class DDIP(BaseDIP):
    def __init__(self, net, input_size=3, lr=0.01, T=2400, schedule="cos", cos_offset=0):
        super().__init__(net, input_size, lr)

        self.T = T
        self.schedule = schedule
        self.cos_offset = cos_offset

    def __str__(self):
        return f"DDIP ({self.schedule}, {self.cos_offset}, {self.T})"
    
    def should_stop(self, state):
        return state["epoch"] >= self.T
    
    def init_z(self, state):
        x = state["x_hat"]
        y = torch.randn_like(state["x_hat"], device=self.device)
        return self._cos_schedule(x, y, 0)

    def update_z(self, z, state):
        x = state["x_out"]
        y = torch.randn_like(x, device=self.device)
        
        if self.schedule == "old":
            z = self._old_schedule(x, y, state["epoch"])
        elif self.schedule == "cos":
            z = self._cos_schedule(x, y, state["epoch"], self.cos_offset)
        elif self.schedule == "linear":
            z = self._linear_schedule(x, y, state["epoch"])

        return z

    def _old_schedule(self, x, y, t):
        alpha_bar = math.cos((math.pi * (self.T - t)) / (2 * (self.T)))**2
        return math.sqrt(alpha_bar)*x + math.sqrt(1 - alpha_bar)*y
    
    def _cos_schedule(self, x, y, t, offset=0):
        alpha_bar = math.cos((math.pi * (self.T - t)) / (2 * (self.T)))**2
        alpha_bar = (1 - 2*offset)*alpha_bar + offset
        return alpha_bar*x + (1 - alpha_bar)*y
    
    def _linear_schedule(self, x, y, t):
        return (t/self.T)*x + (1 - t/self.T)*y
    

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
    

# EXPERIMENTS


class DDIP_AE(DDIP):
    def __init__(self, net, input_size=3, lr=0.01, T=2400, schedule="cos", cos_offset=0):
        super().__init__(net, input_size, lr, T, schedule, cos_offset)

    def __str__(self):
        return f"DDIP AE ({self.schedule}, {self.cos_offset}, {self.T})"
    
    def calculate_loss(self, x, y, state):
        return super().calculate_loss(x, y) + self.mse(x, state["z"])
    

class DDIP_Dual(DDIP_AE):
    def __init__(self, net, input_size=3, lr=0.01, T=2400, schedule="cos", cos_offset=0, k=2):
        self.k = k

        super().__init__(net, input_size, lr, k*T, schedule, cos_offset)

    def __str__(self):
        return f"DDIP Dual k={self.k} ({self.schedule}, {self.cos_offset}, {self.T})"
    
    def optimize(self, input, target, optimizer, state):
        for _ in range(self.k):
            super().optimize(input, target, optimizer, state)


class DDIP_P(DDIP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __str__(self):
        return f"DDIP Prog ({self.schedule}, {self.cos_offset}, {self.T})"

    def on_epoch_end(self, state):
        self.net.update_skip_weights(state["epoch"] / self.T)

        state["metrics"]["skip_weights"] = self.net.skip_weights

        return super().on_epoch_end(state)


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
    

class DDIP_MWV(DIP_MWV, DDIP):
    def __init__(self, net):
        super().__init__(net)

    def __str__(self):
        return "DDIP (MWV)"
    

class DIP_P(DIP):
    def __init__(self, net):
        super().__init__(net)

    def __str__(self):
        return f"DIP Prog"
    
    def on_epoch_end(self, state):
        self.net.update_skip_weights(state["epoch"] / self.max_epochs)

        return super().on_epoch_end(state)

class DDIP_Const(DDIP):
    def __init__(self, net, input_size=3, lr=0.01, T=2400, schedule="cos"):
        super().__init__(net, input_size, lr, T, schedule)

    def __str__(self):
        return f"DDIP Const ({self.schedule}, {self.T})"
    
    def init_z(self, state):
        x = state["x_hat"]
        y = torch.randn_like(state["x_hat"], device=self.device)
        state["noise"] = y
        return self._cos_schedule(x, y, 0)

    def update_z(self, z, state):
        x = state["x_out"]
        y = state["noise"]
        return self._cos_schedule(x, y, state["epoch"])
    
class DIP_Noise(DIP):
    def __init__(self, net, input_size=3, lr=0.01, max_epochs=2400):
        super().__init__(net, input_size, lr, max_epochs)

    def __str__(self):
        return f"DIP Noise"
    
    def init_z(self, state):
        noise = torch.randn_like(state["x_hat"], device=self.device)
        return noise

    def update_z(self, z, state):
        noise = torch.randn_like(state["x_hat"], device=self.device)
        return noise
    