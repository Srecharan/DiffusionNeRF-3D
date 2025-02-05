# src/models/unet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class Attention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        shape = x.shape
        x = x.view(shape[0], self.channels, -1).permute(0, 2, 1)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.permute(0, 2, 1).view(shape)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.double_conv(x)
    
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv_res = nn.Conv2d(channels, channels, 1)  # Added residual connection
        
    def forward(self, x):
        identity = self.conv_res(x)
        h = self.conv1(F.gelu(self.norm1(x)))
        h = self.conv2(F.gelu(self.norm2(h)))
        return h + identity

class FeatureEnhancementBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding='same')  # Changed padding
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=2, dilation=2)  # Adjusted padding
        self.norm2 = nn.GroupNorm(8, channels)
        
    def forward(self, x):
        skip = x
        x = F.gelu(self.norm1(self.conv1(x)))
        x = F.gelu(self.norm2(self.conv2(x)))
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)  # Add this
        return x + skip
    
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, time_emb_dim=256, base_channels=32, attention=True):
        super().__init__()
        
        # Enable gradient checkpointing for memory efficiency
        self.use_checkpointing = True
        self.use_attention = attention
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, 512)  # Reduced from 1024
        )
        
        # Encoder - reduced channel dimensions
        self.enc1 = nn.Sequential(
            DoubleConv(in_channels, 32),  # Reduced from 64
            ResBlock(32)
        )
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = nn.Sequential(
            DoubleConv(32, 64),  # Reduced from 128
            ResBlock(64)
        )
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = nn.Sequential(
            DoubleConv(64, 128),  # Reduced from 256
            ResBlock(128)
        )
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = nn.Sequential(
            DoubleConv(128, 256),  # Reduced from 512
            ResBlock(256)
        )
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            DoubleConv(256, 512),  # Reduced from 1024
            ResBlock(512)
        )
        self.attention = Attention(512)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            DoubleConv(512, 256),
            ResBlock(256),
            FeatureEnhancementBlock(256)
        )

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            DoubleConv(256, 128),
            ResBlock(128),
            FeatureEnhancementBlock(128)
        )

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            DoubleConv(128, 64),
            ResBlock(64),
            FeatureEnhancementBlock(64)
        )

        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            DoubleConv(64, 32),
            ResBlock(32),
            FeatureEnhancementBlock(32)
        )
        
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)
    
    def _run_step(self, x, t_emb, enc_outputs=None):
        # Encoder
        if enc_outputs is None:
            e1 = checkpoint(self.enc1, x) if self.use_checkpointing else self.enc1(x)
            e2 = checkpoint(self.enc2, self.pool1(e1)) if self.use_checkpointing else self.enc2(self.pool1(e1))
            e3 = checkpoint(self.enc3, self.pool2(e2)) if self.use_checkpointing else self.enc3(self.pool2(e2))
            e4 = checkpoint(self.enc4, self.pool3(e3)) if self.use_checkpointing else self.enc4(self.pool3(e3))
            return [e1, e2, e3, e4]
        
        # Decoder
        e1, e2, e3, e4 = enc_outputs
        x = self.attention(x)
        x = x + t_emb
        
        x = self.upconv4(x)
        x = torch.cat([x, e4], dim=1)
        x = checkpoint(self.dec4, x) if self.use_checkpointing else self.dec4(x)
        
        x = self.upconv3(x)
        x = torch.cat([x, e3], dim=1)
        x = checkpoint(self.dec3, x) if self.use_checkpointing else self.dec3(x)
        
        x = self.upconv2(x)
        x = torch.cat([x, e2], dim=1)
        x = checkpoint(self.dec2, x) if self.use_checkpointing else self.dec2(x)
        
        x = self.upconv1(x)
        x = torch.cat([x, e1], dim=1)
        x = checkpoint(self.dec1, x) if self.use_checkpointing else self.dec1(x)
        
        return self.final_conv(x)
        
    def forward(self, x, t):
        # Time embedding
        t = t.float()
        t_emb = self.time_mlp(t.unsqueeze(-1))
        t_emb = t_emb.view(-1, 512, 1, 1)  # Reduced from 1024
        
        # Run encoder
        enc_outputs = self._run_step(x, None)
        
        # Bottleneck
        b = checkpoint(self.bottleneck, self.pool4(enc_outputs[-1])) if self.use_checkpointing else self.bottleneck(self.pool4(enc_outputs[-1]))
        
        # Run decoder
        return self._run_step(b, t_emb, enc_outputs)
    
class DepthPriorModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.prior_encoder = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            ResBlock(channels),
            FeatureEnhancementBlock(channels)
        )
        
    def forward(self, x, prior):
        prior_features = self.prior_encoder(prior)
        return x + prior_features