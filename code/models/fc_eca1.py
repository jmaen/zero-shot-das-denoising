import torch
import torch.nn as nn
import math


# https://github.com/cuiyang512/Unsupervised-DAS-Denoising
class FC(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.in_ch = in_ch
            
        self.fc1 = nn.Sequential(
            nn.Linear(in_ch, out_ch),
            nn.BatchNorm1d(out_ch),
            nn.PReLU(),   
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_ch, out_ch),
            nn.PReLU(),   
        )
        self.fc3 = nn.Sequential(
            nn.Linear(out_ch * 2, out_ch),
            nn.PReLU(),   
        )
        
    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.fc3(x)
        return x


class SE(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.se = nn.Sequential(
            nn.Linear(channels, 1, bias=False),
            nn.ReLU(),
            nn.Linear(1, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.se(x)
        return x * y
    

class Out(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.out = nn.Sequential(
            nn.Linear(in_ch, out_ch // 3),
            # nn.ReLU(),
            nn.Linear(out_ch // 3, out_ch),
        )

    def forward(self, x):
        x = self.out(x)
        return x


class FC_ECA1(nn.Module):
    def __init__(self, channels, freeze=False, label=""):
        super().__init__()

        self.label = label
        
        dims = [channels, 128, 32, 8]

        self.downs = nn.ModuleList([FC(dims[i], dims[i+1]) for i in range(len(dims) - 1)])
        self.ups = nn.ModuleList([FC(dims[-1], dims[-1])] + [FC(dims[i+1] * 2, dims[i]) for i in range(len(dims) - 2, 0, -1)])
        self.skips = nn.ModuleList([SE(dims[i]) for i in range(1, len(dims))])
        self.out = self.out = Out(256, channels)

        if freeze:
            for module in self.downs:
                for param in module.parameters():
                    param.requires_grad = False
            # for module in self.skips:
            #     for param in module.parameters():
            #         param.requires_grad = False

    def __str__(self):
        return f"FC SE 2024 {self.label}"
        
    def forward(self, x):
        C = x.shape[1]
        x = x.flatten(start_dim=1)

        skips = []
        for down, skip in zip(self.downs, self.skips):
            x = down(x)
            skips.append(skip(x))
        
        for up in self.ups:
            x = up(x)
            s = skips.pop()
            x = torch.cat([x, s], dim=1)

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
