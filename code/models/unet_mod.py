import torch
import torch.nn as nn


# TODO: add skip schedules
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
            
        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
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
            nn.BatchNorm2d(out_ch),
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
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2),
        )
        self.convs = ConvBlock(out_ch + skip_ch, out_ch)
        
    def forward(self, x, s):
        x = self.upsample(x)
        if self.skip_ch > 0:
            x = torch.cat([x, s], dim=1)
        x = self.convs(x)
        return x
    

class Skip(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2),
        )
        
    def forward(self, x):
        return self.conv(x)
        

class UNetMod(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, hidden_ch=64, skip_ch=4, n_layers=4, label=""):
        super().__init__()

        self.label = label
        
        down_dims = [in_ch] + [hidden_ch * 2**i for i in range(n_layers)]
        up_dims = [hidden_ch * 2**i for i in range(n_layers, -1, -1)]

        self.downs = nn.ModuleList([Down(down_dims[i], down_dims[i+1]) for i in range(len(down_dims) - 1)])
        self.mid = ConvBlock(down_dims[-1], up_dims[0])
        self.ups = nn.ModuleList([Up(up_dims[i], up_dims[i+1], skip_ch) for i in range(len(up_dims) - 1)])
        self.skips = nn.ModuleList([Skip(down_dims[i], skip_ch) for i in range(1, len(down_dims))])
        self.out = nn.Sequential(
            nn.Conv2d(hidden_ch, out_ch, 1),
            nn.Sigmoid()
        )

    def __str__(self):
        return f"UNetMod {self.label}"
        
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
