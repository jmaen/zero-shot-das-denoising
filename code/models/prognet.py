import torch
import torch.nn as nn


class ProgNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, deep_ch=[8, 16, 32, 64, 128], skip_ch=4, skip_schedules=None, label=""):
        super().__init__()

        self.label = label

        down_ch = [in_ch] + deep_ch[:-1]
        up_ch = deep_ch

        self.downs = nn.ModuleList([self._down(down_ch[i], down_ch[i+1]) for i in range(len(down_ch) - 1)])
        self.mid = self._mid(down_ch[-1], up_ch[-1])
        self.ups = nn.ModuleList([self._up(up_ch[i] + skip_ch, up_ch[i-1]) for i in range(len(up_ch) - 1, 0, -1)])
        self.skips = nn.ModuleList([self._skip(down_ch[i], skip_ch) for i in range(1, len(down_ch))])
        self.out = nn.Sequential(
            nn.Conv2d(up_ch[0], out_ch, kernel_size=1),
            nn.Sigmoid(),
        )

        self.skip_weights = [0 for _ in self.skips]
        self.skip_schedules = skip_schedules or [lambda x: x for _ in self.skips]

    def __str__(self):
        return f"ProgNet ({self.label})"

    def forward(self, x):
        hist = []
        for i, (down, skip) in enumerate(zip(self.downs, self.skips)):
            x = down(x)
            hist.append(self.skip_weights[i] * skip(x))

        x = self.mid(x)

        for up in self.ups:
            x = torch.cat([x, hist.pop()], dim=1)
            x = up(x)

        return self.out(x)

    def update_skip_weights(self, step):
        self.skip_weights = [schedule(step) for schedule in self.skip_schedules]

    def reset_parameters(self, module=None, top_level=True):
        if module is None and top_level:
            for child in self.children():
                self.reset_parameters(child, top_level=False)
        elif isinstance(module, nn.ModuleList) or isinstance(module, nn.Sequential):
            for child in module:
                self.reset_parameters(child, top_level=False)
        elif hasattr(module, 'reset_parameters'):
            module.reset_parameters()  

    def _block(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode='reflect'),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2),
        )

    def _down(self, in_ch, out_ch):
        return nn.Sequential(
            self._block(in_ch, out_ch, stride=2),
            self._block(out_ch, out_ch),
        )

    def _mid(self, in_ch, out_ch):
        return nn.Sequential(
            self._block(in_ch, out_ch),
            self._block(out_ch, out_ch),
        )
    
    def _up(self, in_ch, out_ch):
        return nn.Sequential(
            nn.BatchNorm2d(in_ch),
            self._block(in_ch, out_ch),
            self._block(out_ch, out_ch, kernel_size=1, padding=0),
            nn.Upsample(scale_factor=2, mode='bilinear'),
        ) 

    def _skip(self, in_ch, out_ch):
        return self._block(in_ch, out_ch, kernel_size=1, padding=0)
