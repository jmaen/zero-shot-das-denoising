import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
            
        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2),
        )
        
    def forward(self, x):
        return self.convs(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.convs = ConvBlock(in_ch, out_ch)
        self.downsample = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, stride=2),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2),
        )
        
    def forward(self, x):
        s = self.convs(x)
        x = self.downsample(s)
        return x, s


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch):
        super().__init__()

        self.skip_ch = skip_ch
        
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2),
        )
        self.convs = ConvBlock(out_ch + skip_ch, out_ch)
        
    def forward(self, x, s):
        x = self.upsample(x)
        if self.skip_ch > 0:
            x = torch.cat([x, s], dim=1)
        x = self.convs(x)
        return x
    

class ECA(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()

        kernel_size = int(abs((torch.log2(torch.tensor(channels)) * gamma + b).item())) | 1  # Ensure it's odd
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # (B, C, 1, 1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, 1, c)  # (B, 1, C)
        y = self.conv1d(y).view(b, c, 1, 1)  # (B, C, 1, 1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)
        

class UNetECA(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, hidden_ch=8, n_layers=4, label=""):
        super().__init__()

        self.label = label
        
        down_dims = [in_ch] + [hidden_ch * 2**i for i in range(n_layers)]
        up_dims = [hidden_ch * 2**i for i in range(n_layers, -1, -1)]

        self.downs = nn.ModuleList([Down(down_dims[i], down_dims[i+1]) for i in range(len(down_dims) - 1)])
        self.mid = ConvBlock(down_dims[-1], up_dims[0])
        self.ups = nn.ModuleList([Up(up_dims[i], up_dims[i+1], up_dims[i+1]) for i in range(len(up_dims) - 1)])
        self.skips = nn.ModuleList([ECA(down_dims[i]) for i in range(1, len(down_dims))])
        self.out = nn.Sequential(
            nn.Conv2d(hidden_ch, out_ch, 1),
        )

    def __str__(self):
        return f"UNet ECA {self.label}"
        
    def forward(self, x):
        skips = []
        for down, skip in zip(self.downs, self.skips):
            x, s = down(x)
            skips.append(skip(s))
        
        x = self.mid(x)
        
        for up in self.ups:
            s = skips.pop()
            x = up(x, s)
        
        x = self.out(x)

        return x
    
    def reset_parameters(self, module=None):
        if module is None:
            for child in self.children():
                self.reset_parameters(child)
        elif isinstance(module, nn.ModuleList) or isinstance(module, nn.Sequential):
            for child in module:
                self.reset_parameters(child)
        elif hasattr(module, 'reset_parameters'):
            module.reset_parameters()  
        else:
            for child in module.children():
                self.reset_parameters(child)
