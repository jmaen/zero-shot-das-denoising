import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from .base import Base


class SGDIP(Base):
    def __init__(self, net, loss, early_stopping=False, lr=0.01, max_epochs=2000, k=3, ratio=0.5, use_ref=False, **kwargs):
        super().__init__(net, loss, early_stopping, **kwargs)

        self.lr = lr
        self.max_epochs = max_epochs

        self.k = k
        self.ratio = ratio
        self.use_ref = use_ref

    def __str__(self):
        return f"SGDIP{" - ES" if self.early_stopping else ""} (k={self.k}, r={self.ratio}, ref={self.use_ref})"
    
    def optimize(self, y, state):
        if self.use_ref and "reference" in state and state["reference"] is not None:
            z = state["reference"].clone().detach()

            W, H = z.shape[-2:]
            pad_w = y.shape[-2] - W
            pad_h = y.shape[-1] - H
            pad_l = pad_h // 2
            pad_r = pad_h - pad_l
            pad_t = pad_w // 2
            pad_b = pad_w - pad_t

            z = F.pad(z, (pad_l, pad_r, pad_t, pad_b), mode="reflect")

            z = z.to(self.device).requires_grad_(True)
        else:
            z = torch.randn_like(y, device=self.device, requires_grad=True)

        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        optimizer2 = optim.Adam([z], lr=self.lr)

        for epoch in tqdm(range(self.max_epochs)):
            optimizer.zero_grad()
            optimizer2.zero_grad()
          
            std = z.max() * self.ratio
            x_hat = torch.zeros_like(y, device=self.device)
            for _ in range(self.k):
                noise = torch.randn_like(z, device=self.device) * std
                x_hat += self.net(z + noise)
            x_hat /= self.k
            loss = self.loss(x_hat, y, z, epoch / self.max_epochs)

            loss.backward()
            optimizer.step()
            optimizer2.step()

            state["z"] = z.detach().clone()
            state["x_hat"] = x_hat.detach().clone()
            state["epoch"] = epoch
            state["metrics"]["loss"] = loss.item()
            state["metrics"]["std"] = std.item()
            self.on_epoch_end(state)

            if self.should_stop(state):
                break

        return state["x_hat"]
