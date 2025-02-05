# src/models/feature_extractor.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractor(nn.Module):
    def __init__(self, in_channels=1, feature_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, feature_channels, 3, padding=1),
            nn.GroupNorm(8, feature_channels),
            nn.GELU(),
            nn.Conv2d(feature_channels, feature_channels, 3, padding=1),
            nn.GroupNorm(8, feature_channels),
            nn.GELU(),
        )
        
    def forward(self, x):
        return self.encoder(x)