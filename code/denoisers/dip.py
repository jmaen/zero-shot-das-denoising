import time
from typing import Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim


class BaseDIP:
    def __init__(
        self,
        net: nn.Module,
        input_size: int = 3,
        lr: float = 0.01,
    ):
        self.net = net
        self.input_size = input_size
        self.lr = lr

    def denoise(self, x):
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.net.to(device)

        optimizer = optim.Adam(self.net.parameters(), self.lr)

        x = x.to(device)

        # TODO create init function
        input_shape = [self.input_size if i == 1 else s for i, s in enumerate(x.size())]
        z = torch.rand(input_shape, device=device) * 0.1

        self.net.train()

        print(f"Training on {device}")
        start = time.time()

        state = {"current_epoch": 0, "current_output": None, "losses": []}
        state = self.init_state(state)
        while not self.should_stop(state):
            optimizer.zero_grad()

            out = self.net(z)
            loss = self.calculate_loss(out, x)
            loss.backward()
            optimizer.step()

            state["losses"].append(loss.item())
            state["current_output"] = out
            state = self.update_state(state)
            state["current_epoch"] += 1

            self.handle_logging(state)

        duration = time.time() - start
        print(
            f"Finished training in {time.strftime('%H:%M:%S', time.gmtime(duration))}\n"
        )

        return self.net(z)
    
    def calculate_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Loss function must be implemented by subclass")
    
    def should_stop(self, state: Dict[str, Any]) -> bool:
        raise NotImplementedError("Stopping function must be implemented by subclass")

    def init_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return state

    def update_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return state
    
    def handle_logging(self, state: Dict[str, Any]):
        pass


class DIP(BaseDIP):
    def __init__(self, net, input_size=3, lr=0.01, max_epochs=2400):
        super().__init__(net, input_size, lr)

        self.max_epochs = max_epochs
        self.mse = nn.MSELoss()

    def calculate_loss(self, x, y):
        return self.mse(x, y)
    
    def should_stop(self, state):
        return state["current_epoch"] >= self.max_epochs
    

class DIP_MWV(BaseDIP):
    def __init__(self, net, input_size=3, lr=0.01, window_size=100, patience=1000):
        super().__init__(net, input_size, lr)
         
        self.window_size = window_size
        self.patience = patience
        self.mse = nn.MSELoss()

    def calculate_loss(self, x, y):
        return self.mse(x, y)
    
    def should_stop(self, state):
        if state["current_epoch"] >= state["epoch_opt"] + self.patience:
            print(f"Optimal stopping point: {state["epoch_opt"]} epochs")
            return True
    
    def init_state(self, state):
        state["queue"] = []
        state["var_opt"] = torch.inf
        state["epoch_opt"] = 0
        state["x_opt"] = None

        return state

    def update_state(self, state):
        queue = state["queue"]
        queue.append(state["current_output"])
        if len(queue) > self.window_size:
            queue.pop(0)

            queue = torch.stack(queue)
            var = torch.sum(queue.var(dim=0)).item()

            if var < state["var_opt"]:
                state["var_opt"] = var
                state["epoch_opt"] = state["current_epoch"]
                state["x"] = state["current_output"]

        return state
