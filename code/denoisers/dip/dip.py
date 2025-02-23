import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from .base import Base


class DIP(Base):
    def __init__(self, net, loss, early_stopping=False, lr=0.01, max_epochs=2000, reference=None, **kwargs):
        super().__init__(net, loss, early_stopping)

        self.lr = lr
        self.max_epochs = max_epochs
        self.reference = reference

    def __str__(self):
        return f"DIP{" - ES" if self.early_stopping else ""}"

    def optimize(self, y, state):
        if self.reference is not None:
            z = self.reference.clone().detach()

            W, H = z.shape[-2:]
            pad_w = y.shape[-2] - W
            pad_h = y.shape[-1] - H
            z = F.pad(z, (0, pad_h, 0, pad_w), mode="constant", value=0)

            z = z.to(self.device).requires_grad_(True)
        else:
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
