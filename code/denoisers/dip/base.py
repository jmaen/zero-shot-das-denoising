import time
from typing import Any, Dict, Literal, TypedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Logger
from ..denoiser import Denoiser
import matplotlib.pyplot as plt


class LoggingOptions(TypedDict):
    mode: Literal["local", "wandb"]
    config: Dict[str, str]
    metrics: Dict[str, nn.Module]
    log_output: bool


class Base(Denoiser):
    def __init__(self, net: nn.Module, loss: nn.Module, early_stopping: bool = False, **kwargs):
        self.device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.net = net
        self.net.to(self.device)
        self.loss = loss

        self.early_stopping = early_stopping
        if self.early_stopping:
            self.patience = 500
            self.window_size = 100

        self.logger = Logger()

    def name(self):
        return f"{str(self)} | {str(self.loss)} | {str(self.net)}"

    def denoise(self, y: torch.Tensor, x: torch.Tensor = None, logging_options: LoggingOptions = None, init_state: Dict[str, Any] = None):
        # logging
        if logging_options is None:
            logging_options = {
                "mode": "local",
                "config": {
                    "data_id": 0,
                    "denoiser_id": 0,
                    "out_dir": "",
                },
                "metrics": {},
                "log_output": False,
            }

        self.metrics = {key: metric.to(self.device) for key, metric in logging_options["metrics"].items()}
        self.log_output = logging_options["log_output"]

        logging_options["config"].update({
            "variant": str(self),
            "loss": str(self.loss),
            "architecture": str(self.net),
        })

        self.logger.init_run(logging_options["mode"], logging_options["config"])

        # padding
        W, H = y.shape[-2:]
        pad_w = (1 << (W - 1).bit_length()) - W
        pad_h = (1 << (H - 1).bit_length()) - H
        pad_l = pad_h // 2
        pad_r = pad_h - pad_l
        pad_t = pad_w // 2
        pad_b = pad_w - pad_t

        y = F.pad(y, (pad_l, pad_r, pad_t, pad_b), mode="reflect")
        y = y.to(self.device)
        if x is not None:
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b), mode="reflect")
            x = x.to(self.device)

        # training
        state = {
            "y": y,
            "x": x, 
            "x_hat": None,
            "z": None,
            "start": time.time(),
            "epoch": 0, 
            "metrics": {},
            "summary": {},
        }
        if init_state is not None:
            state.update(init_state)

        self.on_train_start(state)

        x_hat = self.optimize(y, state)

        logs = self.on_train_end(state)

        self.net.reset_parameters()

        # unpadding
        x_hat = x_hat[:, :, pad_t:W+pad_t, pad_l:H+pad_l]

        return x_hat, state, logs
    
    def optimize(self, y: torch.Tensor, state: Dict[str, Any] = {}):
        raise NotImplementedError("Optimization function must be implemented by subclass")
    
    def should_stop(self, state):
        if self.early_stopping and state["epoch"] >= state["epoch_opt"] + self.patience:
            state["x_hat"] = state["x_opt"]
            return True

        return False
    
    def on_train_start(self, state):
        if self.early_stopping:
            state["queue"] = []
            state["var_opt"] = torch.inf
            state["epoch_opt"] = 0
            state["x_opt"] = None

    def on_epoch_end(self, state: Dict[str, Any]):
        if self.early_stopping:       
            queue = state["queue"]
            queue.append(state["x_hat"])
            if len(queue) > self.window_size:
                queue.pop(0)

                queue = torch.stack(queue)
                var = torch.sum(queue.var(dim=0)).item()

                state["metrics"]["var"] = var

                if var < state["var_opt"]:
                    state["var_opt"] = var
                    state["epoch_opt"] = state["epoch"]
                    state["x_opt"] = state["x_hat"]

        if state["x"] is not None:
            for key, metric in self.metrics.items():
                result = metric(state["x_hat"], state["x"]).item()
                state["metrics"][key] = result

        if self.log_output and state["epoch"] % 10 == 0:
            output = [state["y"], state["x_hat"]]
            if state["z"] is not None:
                output.insert(1, state["z"])
            if state["x"] is not None:
                output.append(state["x"])
            state["metrics"]["output"] = output

        self.logger.log(state["metrics"])
    
    def on_train_end(self, state: Dict[str, Any]):
        duration = time.time() - state["start"]
        runtime = time.strftime('%H:%M:%S', time.gmtime(duration))
        state["summary"]["runtime"] = runtime

        if self.early_stopping:
            self.logger.mark(state["epoch_opt"])
            state["summary"]["stopping_point"] = state["epoch_opt"]

        if state["x"] is not None:
            for key, metric in self.metrics.items():
                result = metric(state["x_hat"], state["x"]).item()
                state["summary"][key] = result

        data = self.logger.finish(state["summary"])

        return data
