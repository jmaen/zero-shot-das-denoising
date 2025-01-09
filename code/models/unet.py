import torch
import torch.nn as nn

# https://arxiv.org/abs/1505.04597

# TODO: add weight reset
class ConvBlock(nn.Module):
    
    def __init__(self, in_ch, out_ch, h_ch=None):
        super().__init__()
        if h_ch is None:
            h_ch = out_ch
            
        self.layers = nn.Sequential(
            nn.Conv2d(in_ch, h_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(h_ch),
            nn.ReLU(),
            nn.Conv2d(h_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )
        
    def forward(self, x):
        return self.layers(x)
    
class Down(nn.Module):
    
    def __init__(self, in_ch, out_ch, h_ch=None, factor=2):
        super().__init__()
        self.convs = ConvBlock(in_ch, out_ch, h_ch)
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
                nn.ReLU(),
            )
    
        self.convs = ConvBlock(in_ch, out_ch)
        
    def forward(self, x, h):
        x = self.upsample(x)
        x = torch.cat([x, h], dim=1)
        x = self.convs(x)
        return x
        
class Mid(nn.Module):
    
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.convs = ConvBlock(in_ch, out_ch)
        
    def forward(self, x):
        x = self.convs(x)
        return x
        
class UNet(nn.Module):
        
    
    def __init__(self, in_ch=3, out_ch=3, hidden_ch=64, n_layers=4, factor=2, bilinear=True):
        super().__init__()
        
        down_dims = [in_ch] + [hidden_ch*2**i for i in range(n_layers)]
        up_dims = [hidden_ch*2**i for i in range(n_layers,-1,-1)]
        self.downs = nn.ModuleList([Down(down_dims[i], down_dims[i+1], factor=factor) for i in range(len(down_dims) - 1)])
        
        self.mid = Mid(down_dims[-1], down_dims[-1]*2)
        self.ups = nn.ModuleList([Up(up_dims[i], up_dims[i+1], factor=factor, bilinear=bilinear) for i in range(len(up_dims) - 1)])
        
        self.outconv = nn.Sequential(
            nn.Conv2d(hidden_ch, out_ch, 1),
            nn.Sigmoid()
        )

    def __str__(self):
        return "UNet"
        
    def forward(self, x):
        
        hist = []
        for down in self.downs:
            x, h = down(x)
            hist.append(h)
        
        x = self.mid(x)
        
        for up in self.ups:
            h = hist.pop()
            x = up(x, h)
        
        x = self.outconv(x)
        
        return x