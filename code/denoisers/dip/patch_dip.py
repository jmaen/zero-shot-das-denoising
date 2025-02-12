import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from .base import Base
from utils.patch import *


class PatchDIP(Base):
    def __init__(self, net, loss, epochs=50, batch_size=1024, lr=0.01, kurtosis_threshold=0.05, use_diff=False, **kwargs):
        super().__init__(net, loss, False)

        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.kurtosis_threshold = kurtosis_threshold
        self.use_diff = use_diff

    def __str__(self):
        return f"PatchDIP (epochs={self.epochs})"

    def optimize(self, y, state):
        shape = y.shape[-2:]

        y_patches = patch(y)
        y_filtered_patches = filter_patches(y_patches, self.kurtosis_threshold)

        train_loader = DataLoader(TensorDataset(y_filtered_patches), batch_size=self.batch_size, shuffle=True, )
        pred_loader = DataLoader(TensorDataset(y_patches), batch_size=self.batch_size)

        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

        num_batches = len(train_loader)  
        train_batches = int(0.9 * num_batches)

        self.net.train()
        for epoch in tqdm(range(self.epochs)):
            for i, (batch,) in enumerate(train_loader):
                if i > train_batches:
                    break

                batch = batch.to(self.device)

                optimizer.zero_grad()

                out = self.net(batch)
                loss = self.loss(out, batch)

                loss.backward()
                optimizer.step()

            # x_hat = self.predict(pred_loader, shape, y)
            # state["x_hat"] = x_hat.detach().clone()
            # state["epoch"] = epoch
            # self.on_epoch_end(state)

        x_hat = self.predict(pred_loader, shape, y)
        state["x_hat"] = x_hat.detach().clone()

        return state["x_hat"]
    
    def predict(self, pred_loader, shape, y):
        x_hat_patches = []
        with torch.no_grad():
            for batch, in pred_loader:
                batch = batch.to(self.device)
                out = self.net(batch)
                x_hat_patches.append(out)

        x_hat_patches = torch.cat(x_hat_patches, dim=0)
        x_hat = unpatch(x_hat_patches, shape)

        if self.use_diff:
            x_hat = y - x_hat

        return x_hat