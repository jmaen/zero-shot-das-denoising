import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from .unet import UNet


class DIP:
  def __init__(self, input_channels, deep_channels, skip_channels, epochs=1800, lr=0.01):
    self.input_channels = input_channels
    self.deep_channels = deep_channels
    self.skip_channels = skip_channels
    self.epochs = epochs
    self.lr = lr

  def denoise(self, x0):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    net = UNet(self.input_channels, self.deep_channels, self.skip_channels)
    net.to(device)

    optimizer = optim.Adam(net.parameters(), self.lr)
    mse = nn.MSELoss()

    x0 = x0.to(device)

    input_shape = [self.input_channels if i == 1 else s for i, s in enumerate(x0.size())]
    z = torch.rand(input_shape, device=device) * 0.1  # TODO regularize noise?
    
    net.train()

    print(f'Training on {device}')
    print('----------')
    start = time.time()

    losses = []
    for i in range(self.epochs):
      optimizer.zero_grad()

      out = net(z)
      loss = mse(out, x0)
      loss.backward()
      optimizer.step()

      losses.append(loss.item())

      if i % 100 == 99:
        print(f'Epoch {i + 1:4d}/{self.epochs} | Loss: {loss.item()}')

    duration = time.time() - start
    print('----------')
    print(f'Finished training in {time.strftime('%H:%M:%S', time.gmtime(duration))}')

    plt.plot(losses)
    plt.show()

    return net(z)
