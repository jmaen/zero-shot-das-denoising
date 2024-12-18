import torch
import torch.nn as nn


class SkipNet(nn.Module):
  def __init__(self, in_channels, deep_channels, skip_channels, out_channels=3):
    super().__init__()

    assert len(deep_channels) == len(skip_channels)

    self.deep_channels = deep_channels
    self.skip_channels = skip_channels
    self.depth = len(deep_channels)
    
    down = [None for _ in range(self.depth)]
    up = [None for _ in range(self.depth)]
    skip = [None for _ in range(self.depth)]

    # first down & skip layers
    down[0] = self._down(in_channels, deep_channels[0])

    if skip_channels[0] > 0:
      skip[0] = self._skip(in_channels, skip_channels[0])

    for i in range(self.depth - 1):
      # down layers
      ic = deep_channels[i]
      oc = deep_channels[i + 1]
      is_last = i == self.depth - 2
      down[i + 1] = self._down(ic, oc, is_last)

      # up layers
      ic = deep_channels[i + 1] + skip_channels[i]
      oc = deep_channels[i]
      is_first = i == 0
      up[i] = self._up(ic, oc, is_first)

      # skip layers
      if skip_channels[i + 1] > 0:
        ic = deep_channels[i]
        oc = skip_channels[i + 1]
        skip[i + 1] = self._skip(ic, oc)

    # last ("deepest") up layer
    ic = deep_channels[self.depth - 1] + skip_channels[self.depth - 1]
    oc = deep_channels[self.depth - 1]
    up[-1] = self._up(ic, oc)

    self.down = nn.ModuleList(down)
    self.up = nn.ModuleList(up)
    self.skip = nn.ModuleList(skip)

    self.out = nn.Sequential(
      nn.Conv2d(deep_channels[0], out_channels, kernel_size=1),
      nn.Sigmoid()
    )

  def __str__(self):
    return f"SkipNet ({str(self.deep_channels)}, {str(self.skip_channels)})"

  def forward(self, x):
    s = [None for _ in range(self.depth)]

    # calculate down and skip outputs
    for i in range(0, self.depth):
      if self.skip[i] is not None:
        s[i] = self.skip[i](x)
    
      x = self.down[i](x)

    # calculate up outputs
    for i in range(self.depth - 1, -1, -1):
      x = self._concat(x, s[i])
      x = self.up[i](x)

    return self.out(x)
  
  def _down(self, in_channels, out_channels, is_last=False):
    model = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, padding_mode='reflect'),
      nn.BatchNorm2d(out_channels),
      nn.LeakyReLU(0.2),
      nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
      nn.BatchNorm2d(out_channels),
      nn.LeakyReLU(0.2),
    )

    if is_last:
      model.append(nn.Upsample(scale_factor=2, mode='bilinear'))

    return model

  def _up(self, in_channels, out_channels, is_first=False):
    model = nn.Sequential(
      nn.BatchNorm2d(in_channels),
      nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect'),
      nn.BatchNorm2d(out_channels),
      nn.LeakyReLU(0.2),
      nn.Conv2d(out_channels, out_channels, kernel_size=1),
      nn.BatchNorm2d(out_channels),
      nn.LeakyReLU(0.2),
    )

    if not is_first:
      model.append(nn.Upsample(scale_factor=2, mode='bilinear'))

    return model
  
  def _skip(self, in_channels, out_channels):
    return nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size=1),
      nn.BatchNorm2d(out_channels),
      nn.LeakyReLU(0.2),
    )
  
  def _concat(self, x, s):
    if s is None:
      return x
    
    return torch.cat([x, s], dim=1)
