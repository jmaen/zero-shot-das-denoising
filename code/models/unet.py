import torch
import torch.nn as nn


class UNet(nn.Module):
  def __init__(self, in_channels, deep_channels=[8, 16, 32, 64, 128], skip_channels=[0, 0, 0, 4, 4]):
    super().__init__()

    self.down1 = self._down(in_channels, 8)
    self.down2 = self._down(8, 16)
    self.down3 = self._down(16, 32)
    self.down4 = self._down(32, 64)
    self.down5 = self._down(64, 128, True)

    self.skip4 = self._skip(32, 4)
    self.skip5 = self._skip(64, 4)

    self.up1 = self._up(16, 8, True)
    self.up2 = self._up(32, 16)
    self.up3 = self._up(64, 32)
    self.up4 = self._up(128 + 4, 64)
    self.up5 = self._up(128 + 4, 128)

    self.out = nn.Sequential(
      nn.Conv2d(8, 3, kernel_size=1),
      nn.Sigmoid()
    )

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

  def forward(self, x):
    x1 = self.down1(x)
    x2 = self.down2(x1)
    x3 = self.down3(x2)
    x4 = self.down4(x3)
    x5 = self.down5(x4)

    s4 = self.skip4(x3)
    s5 = self.skip5(x4)

    u5 = self.up5(torch.cat([x5, s5], dim=1))
    u4 = self.up4(torch.cat([u5, s4], dim=1))
    u3 = self.up3(u4)
    u2 = self.up2(u3)
    u1 = self.up1(u2)

    return self.out(u1)
