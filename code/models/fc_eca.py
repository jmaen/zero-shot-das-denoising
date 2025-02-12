import torch
import torch.nn as nn
import math


# https://github.com/omarmohamed15/DIP_for_3D_Seismic_Denoising
class FC(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.in_ch = in_ch
            
        self.fc = nn.Sequential(
            nn.Linear(in_ch, out_ch),
            nn.ELU(),
        )
        
    def forward(self, x):
        return self.fc(x)


class SE(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.se = nn.Sequential(
            nn.Linear(channels, channels // 16, bias=False),
            nn.ReLU(),
            nn.Linear(channels // 16, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.se(x)
        return x * y


class FC_ECA(nn.Module):
    def __init__(self, channels, freeze=False, label=""):
        super().__init__()

        self.label = label
        
        dims = [channels, 128, 64, 32, 16, 8]

        self.downs = nn.ModuleList([FC(dims[i], dims[i+1]) for i in range(len(dims) - 1)])
        self.ups = nn.ModuleList([FC(dims[i+1] * 2, dims[i]) for i in range(len(dims) - 2, -1, -1)])
        self.skips = nn.ModuleList([SE(dims[i+1]) for i in range(len(dims) - 1)])
        self.out = nn.Linear(channels, channels)

        if freeze:
            for module in self.downs:
                for param in module.parameters():
                    param.requires_grad = False
            # for module in self.skips:
            #     for param in module.parameters():
            #         param.requires_grad = False

    def __str__(self):
        return f"FC SE {self.label}"
        
    def forward(self, x):
        C = x.shape[1]
        x = x.flatten(start_dim=1)

        skips = []
        for down, skip in zip(self.downs, self.skips):
            x = down(x)
            skips.append(skip(x))
        
        for up in self.ups:
            s = skips.pop()
            x = torch.cat([x, s], dim=1)
            x = up(x)

        x = self.out(x)

        x = x.unflatten(dim=1, sizes=(C, math.isqrt(x.shape[1] // C), -1))

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
