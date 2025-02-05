import torch
import torch.nn as nn
import torch.nn.functional as F

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, temb_channels=512, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)
        
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        if self.in_channels != self.out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x, temb=None):
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)
        
        if temb is not None:
            h = h + self.temb_proj(F.silu(temb))[:, :, None, None]
            
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        if self.in_channels != self.out_channels:
            x = self.shortcut(x)
            
        return x + h

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels=512, dropout=0.0):
        super().__init__()
        self.resblocks = nn.ModuleList([
            ResnetBlock(in_channels, out_channels, temb_channels, dropout),
            ResnetBlock(out_channels, out_channels, temb_channels, dropout)
        ])
        self.downsample = nn.Conv2d(out_channels, out_channels, 4, 2, 1)
        
    def forward(self, x, temb=None):
        for resblock in self.resblocks:
            x = resblock(x, temb)
        return self.downsample(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels=512, dropout=0.0):
        super().__init__()
        self.resblocks = nn.ModuleList([
            ResnetBlock(in_channels, out_channels, temb_channels, dropout),
            ResnetBlock(out_channels, out_channels, temb_channels, dropout)
        ])
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        
    def forward(self, x, temb=None):
        x = self.upsample(x)
        for resblock in self.resblocks:
            x = resblock(x, temb)
        return x