import time
import math
from typing import Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import wandb
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
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

        self.psnr = PeakSignalNoiseRatio().to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure().to(self.device)

    def denoise(self, x_hat, x=None, id=0):
        self.net.to(self.device)

        optimizer = optim.Adam(self.net.parameters(), self.lr)

        x_hat = x_hat.to(self.device)
        if x is not None:
            x = x.to(self.device)

        state = {
            "id": id,
            "x_hat": x_hat,
            "x": x, 
            "x_out": None,
            "start": time.time(),
            "epoch": 0, 
            "metrics": {}
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
    
    def calculate_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Loss function must be implemented by subclass")
    
    def should_stop(self, state: Dict[str, Any]) -> bool:
        raise NotImplementedError("Stopping function must be implemented by subclass")
    
    def init_z(self, state: Dict[str, Any]) -> torch.Tensor:
        return torch.rand_like(state["x_hat"], device=self.device) * 0.1
    
    def update_z(self, z: torch.Tensor, state: Dict[str, Any]) -> torch.Tensor:
        return z

    def on_train_start(self, state: Dict[str, Any]):
        print(f"Training on {self.device}")

        wandb.init(
            project="zero-shot-das-denoising",
            entity="jmaen-team",
            name=f"{state["id"]} | {str(self)} | {str(self.net)}",
            settings=wandb.Settings(init_timeout=120),
            config={
                "variant": str(self),
                "architecture": str(self.net),
            }
        )

    def on_epoch_end(self, state: Dict[str, Any]):
        psnr = self.psnr(state["x_out"], state["x"]).item()
        ssim = self.ssim(state["x_out"], state["x"]).item()

        state["metrics"]["psnr"] = psnr
        state["metrics"]["ssim"] = ssim

        wandb.log(state["metrics"])
    
    def on_train_end(self, state: Dict[str, Any]):
        psnr = self.psnr(state["x_out"], state["x"]).item()
        ssim = self.ssim(state["x_out"], state["x"]).item()

        wandb.run.summary["out_psnr"] = psnr
        wandb.run.summary["out_ssim"] = ssim
        wandb.finish()

        duration = time.time() - state["start"]
        print(
            f"Finished training in {time.strftime('%H:%M:%S', time.gmtime(duration))}\n"
        )

class DIP(BaseDIP):
    def __init__(self, net, input_size=3, lr=0.01, max_epochs=2400):
        super().__init__(net, input_size, lr)

        self.max_epochs = max_epochs
        self.mse = nn.MSELoss()

    def __str__(self):
        return f"DIP ({self.max_epochs})"

    def calculate_loss(self, x, y):
        return self.mse(x, y)
    
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

    def calculate_loss(self, x, y):
        return self.mse(x, y)
    
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
        wandb.run.summary["stopping_point"] = state["epoch_opt"]

        print(f"Early stopping point: {state["epoch_opt"]} epochs")

        super().on_train_end(state)

        # fig, ax1 = plt.subplots(figsize=(8, 5))
        # ax1.grid()

        # ax1.plot(state["losses"], 'b-')
        # ax1.set_xlabel('Epoch')
        # ax1.set_ylabel('Loss', color='b')
        # ax1.tick_params(axis='y', labelcolor='b')

        # ax2 = ax1.twinx()
        # ax2.plot(state["psnrs"], 'r-')
        # ax2.set_ylabel('PSNR', color='r')
        # ax2.tick_params(axis='y', labelcolor='r')

        # ax3 = ax1.twinx()
        # ax3.spines['right'].set_position(('outward', 60))
        # ax3.plot(state["vars"], 'g-')
        # ax3.set_ylabel('Variance', color='g')
        # ax3.tick_params(axis='y', labelcolor='g')

        # ax1.axvline(x=state["epoch_opt"], color='purple', linestyle='--')

        # fig.tight_layout()
        # plt.show()


class DIP_TV(BaseDIP):
    def __init__(self, net, input_size=3, lr=0.01, max_epochs=2400):
        super().__init__(net, input_size, lr)

        self.max_epochs = max_epochs
        self.mse = nn.MSELoss()

    def __str__(self):
        return f"DIP-TV ({self.max_epochs})"

    def calculate_loss(self, x, y):
        return self.mse(x, y) + self._tv_norm(x)
    
    def should_stop(self, state):
        return state["epoch"] >= self.max_epochs

    def _tv_norm(self, x):
        diff_v = torch.abs(x[:, :-1, :] - x[:, 1:, :])
        diff_h = torch.abs(x[:, :, :-1] - x[:, :, 1:])

        tv_norm = torch.sum(diff_v) + torch.sum(diff_h)

        normalized = tv_norm / x.sum()

        return normalized.item()


class DDIP(BaseDIP):
    def __init__(self, net, input_size=3, lr=0.01, max_epochs=2800):
        super().__init__(net, input_size, lr)

        self.max_epochs = max_epochs
        self.mse = nn.MSELoss()

    def __str__(self):
        return f"DDIP ({self.max_epochs})"

    def calculate_loss(self, x, y):
        return self.mse(x, y)
    
    def should_stop(self, state):
        return state["epoch"] >= self.max_epochs
    
    def init_z(self, state):
        x = state["x_hat"]
        y = torch.randn_like(state["x_hat"], device=self.device)
        return self._cos_schedule(x, y, 0)

    def update_z(self, z, state):
        x = state["x_out"]
        y = torch.randn_like(state["x_out"], device=self.device)
        return self._cos_schedule(x, y, state["epoch"])

    def _cos_schedule(self, x, y, t):
        alpha = math.cos((math.pi * (self.max_epochs - t)) / (2 * (self.max_epochs + 200)))**2
        return math.sqrt(alpha) * x + math.sqrt(1 - alpha) * y
