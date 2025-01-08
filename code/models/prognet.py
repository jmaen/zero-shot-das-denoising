import torch
import torch.nn as nn


class ProgNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, deep_ch=[8, 16, 32, 64, 128], skip_ch=4):
        super().__init__()

        down_ch = [in_ch] + deep_ch[:-1]
        up_ch = deep_ch

        self.downs = nn.ModuleList([Down(down_ch[i], down_ch[i+1]) for i in range(len(down_ch) - 1)])
        self.mid = Mid(down_ch[-1], up_ch[-1])
        self.ups = nn.ModuleList([Up(up_ch[i] + skip_ch, up_ch[i-1]) for i in range(len(up_ch) - 1, 0, -1)])
        self.skips = nn.ModuleList([Skip(down_ch[i], skip_ch) for i in range(1, len(down_ch))])
        self.out = nn.Sequential(
            nn.Conv2d(up_ch[0], out_ch, kernel_size=1),
            nn.Sigmoid(),
        )

        self.skip_weight = 0

    def __str__(self):
        return "ProgNet"

    def forward(self, x):
        hist = []
        for i, (down, skip) in enumerate(zip(self.downs, self.skips)):
            x = down(x)
            hist.append(self.skip_weight * skip(x))

        x = self.mid(x)

        for up in self.ups:
            x = torch.cat([x, hist.pop()], dim=1)
            x = up(x)

        return self.out(x)

    def update_skip_weight(self, weight):
        self.skip_weight = weight
    

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode='reflect'),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.block = nn.Sequential(
            Block(in_ch, out_ch, stride=2),
            Block(out_ch, out_ch),
        )

    def forward(self, x):
        return self.block(x)
    

class Mid(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.block = nn.Sequential(
            Block(in_ch, out_ch),
            Block(out_ch, out_ch),
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.block = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            Block(in_ch, out_ch),
            Block(out_ch, out_ch, kernel_size=1, padding=0),
            nn.Upsample(scale_factor=2, mode='bilinear'),
        )

    def forward(self, x):
        return self.block(x)
    

class Skip(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.block = Block(in_ch, out_ch, kernel_size=1, padding=0)

    def forward(self, x):
        return self.block(x)
