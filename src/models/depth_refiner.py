import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleDepthRefiner(nn.Module):
    def __init__(self, in_channels=1, base_channels=32):
        super().__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, base_channels*2, 3, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        self.final = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, in_channels, 1),
            nn.Tanh()  # Since input is normalized to [-1,1]
        )
        
        # Edge detection kernels
        self.register_buffer('sobel_x', torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ]).reshape(1, 1, 3, 3).float())
        
        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ]).reshape(1, 1, 3, 3).float())
        
    def compute_edges(self, x):
        # Pad input for same size output
        x_pad = F.pad(x, (1, 1, 1, 1), mode='reflect')
        
        # Apply Sobel filters
        gx = F.conv2d(x_pad, self.sobel_x)
        gy = F.conv2d(x_pad, self.sobel_y)
        
        return torch.sqrt(gx**2 + gy**2)
        
    def forward(self, x):
        # Compute edges for skip connection
        edges = self.compute_edges(x)
        
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        
        # Decoder with skip connections
        d1 = F.interpolate(e2, size=e1.shape[2:], mode='bilinear', align_corners=True)
        d1 = self.dec1(d1)
        d1 = torch.cat([d1, e1], dim=1)
        
        # Final output with edge preservation
        out = self.final(d1)
        out = out + 0.1 * edges * x  # Edge-aware residual connection
        
        return out

    def loss_fn(self, pred, target):
        # L1 loss
        l1_loss = F.l1_loss(pred, target)
        
        # Edge loss
        pred_edges = self.compute_edges(pred)
        target_edges = self.compute_edges(target)
        edge_loss = F.mse_loss(pred_edges, target_edges)
        
        # Total variation for smoothness
        tv_loss = torch.mean(torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])) + \
                 torch.mean(torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :]))
        
        return l1_loss + 0.1 * edge_loss + 0.01 * tv_loss