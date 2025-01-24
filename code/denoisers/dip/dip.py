import torch
import torch.optim as optim
from tqdm import tqdm
from .base import Base


class DIP(Base):
    def __init__(self, net, loss, early_stopping=False, lr=0.01, max_epochs=2000, **kwargs):
        super().__init__(net, loss, early_stopping)

        self.lr = lr
        self.max_epochs = max_epochs

    def __str__(self):
        return f"DIP{" - ES" if self.early_stopping else ""}"

    def optimize(self, y, state):
        z = torch.randn_like(y, device=self.device)

        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

        for epoch in tqdm(range(self.max_epochs)):
            optimizer.zero_grad()

            x_hat = self.net(z)
            loss = self.loss(x_hat, y, z, epoch / self.max_epochs)
            
            loss.backward()
            optimizer.step()

            state["z"] = z.detach().clone()
            state["x_hat"] = x_hat.detach().clone()
            state["epoch"] = epoch
            state["metrics"]["loss"] = loss.item()
            self.on_epoch_end(state)

            if self.should_stop(state):
                break

        return state["x_hat"]
