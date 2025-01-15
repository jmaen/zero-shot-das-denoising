import torch
import torch.nn as nn

# https://arxiv.org/abs/1505.04597

class ConvBlock(nn.Module):
    
    def __init__(self, in_ch, out_ch, h_ch=None):
        super().__init__()
        if h_ch is None:
            h_ch = out_ch
            
        self.layers = nn.Sequential(
            nn.Conv2d(in_ch, h_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(h_ch),
            nn.LeakyReLU(0.2),
            nn.Conv2d(h_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2),
        )
        
    def forward(self, x):
        return self.layers(x)
    
class Down(nn.Module):
    
    def __init__(self, in_ch, out_ch, factor=2):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2),
        )
        self.downsample = nn.MaxPool2d(factor)
        
    def forward(self, x):
        h = self.convs(x)
        x = self.downsample(h)
        return x, h
    
class Up(nn.Module):
    
    def __init__(self, in_ch, out_ch, factor=2, bilinear=True):
        super().__init__()
        
        self.upsample = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=factor, stride=factor)
        if bilinear:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=factor, mode='bilinear'),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2),
            )
    
        self.convs = nn.Sequential(
            nn.BatchNorm2d(out_ch + 4),
            nn.Conv2d(out_ch + 4, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_ch, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2),
        )
        
    def forward(self, x, h):
        x = self.upsample(x)
        x = torch.cat([x, h], dim=1)
        x = self.convs(x)
        return x
    

class Skip(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2),
        )
        
    def forward(self, x):
        x = self.convs(x)
        return x
        
class UNet3(nn.Module):
        
    
    def __init__(self, in_ch=3, out_ch=3, hidden_ch=64, n_layers=4, factor=2, bilinear=True):
        super().__init__()
        
        down_dims = [in_ch] + [hidden_ch*2**i for i in range(n_layers)]
        up_dims = [hidden_ch*2**i for i in range(n_layers,-1,-1)]

        self.downs = nn.ModuleList([Down(down_dims[i], down_dims[i+1], factor=factor) for i in range(len(down_dims) - 1)])
        self.skips = nn.ModuleList([Skip(down_dims[i], 4) for i in range(1, len(down_dims))])
        self.mid = ConvBlock(down_dims[-1], down_dims[-1]*2)
        self.ups = nn.ModuleList([Up(up_dims[i], up_dims[i+1], factor=factor, bilinear=bilinear) for i in range(len(up_dims) - 1)])
        
        self.outconv = nn.Sequential(
            nn.Conv2d(hidden_ch, out_ch, 1),
            nn.Sigmoid()
        )

    def __str__(self):
        return "UNet3"
        
    def forward(self, x):
        
        hist = []
        for down, skip in zip(self.downs, self.skips):
            x, h = down(x)
            hist.append(skip(h))
        
        x = self.mid(x)
        
        for up in self.ups:
            h = hist.pop()
            x = up(x, h)
        
        x = self.outconv(x)
        
        return x
    
    def reset_parameters(self, module=None, top_level=True):
        if module is None:
            if top_level:
                for child in self.children():
                    self.reset_parameters(child, top_level=False)
        elif isinstance(module, nn.ModuleList) or isinstance(module, nn.Sequential):
            for child in module:
                self.reset_parameters(child, top_level=False)
        elif hasattr(module, 'reset_parameters'):
            module.reset_parameters()  
        else:
            for child in module.children():
                self.reset_parameters(child)
