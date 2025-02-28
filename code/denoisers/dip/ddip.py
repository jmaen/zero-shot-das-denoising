import math
import torch
import torch.optim as optim
from tqdm import tqdm
from .base import Base
from .schedules import Schedule


class DDIP(Base):
    def __init__(self, net, loss, schedule: Schedule, early_stopping=False, lr=0.01, max_epochs=2000, normalize=True, k=1, **kwargs):
        super().__init__(net, loss, early_stopping, **kwargs)

        self.lr = lr
        self.max_epochs = max_epochs

        self.schedule = schedule
        self.normalize = normalize
        self.k = k

    def __str__(self):
        return f"DDIP{" - ES" if self.early_stopping else ""} ({str(self.schedule)})"
    
    def optimize(self, y, state):
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

        x_hat = y   # use noisy signal as initialization
        for epoch in tqdm(range(self.max_epochs)):
            optimizer.zero_grad()

            alpha_bar = self.schedule(epoch / self.max_epochs)

            x_hat_next = torch.zeros_like(y, device=self.device)
            for _ in range(self.k):
                z = self._forward(x_hat, alpha_bar)
                x_hat_next += self.net(z)
            x_hat = x_hat_next / self.k
            loss = self.loss(x_hat, y, z, epoch / self.max_epochs)

            loss.backward()
            optimizer.step()

            state["z"] = z.detach().clone()
            state["x_hat"] = x_hat.detach().clone()
            state["epoch"] = epoch
            state["metrics"]["loss"] = loss.item()
            state["metrics"]["alpha_bar"] = alpha_bar
            self.on_epoch_end(state)

            if self.should_stop(state):
                break

        return state["x_hat"]

    def _forward(self, x_hat, alpha_bar):
        if self.normalize:
            z = math.sqrt(alpha_bar)*x_hat.detach() + math.sqrt(1 - alpha_bar)*torch.randn_like(x_hat, device=self.device)
        else:
            z = alpha_bar*x_hat.detach() + (1 - alpha_bar)*torch.randn_like(x_hat, device=self.device)

        return z
