# src/models/diffusion.py

import torch
import torch.nn.functional as F
import torchvision.models as models
import torch.nn as nn
from src.models.feature_extractor import FeatureExtractor

class DiffusionModel:
    def __init__(self, model, n_steps=1000, beta_start=1e-4, beta_end=0.02, beta_schedule='linear', device='cuda'):
        self.model = model
        self.n_steps = n_steps
        self.device = device
        self.config = config
        
        if beta_schedule == 'cosine':
            self.betas = self._cosine_beta_schedule(n_steps)
        else:
            self.betas = torch.linspace(beta_start, beta_end, n_steps)
            
        self.betas = self.betas.to(device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.perceptual_loss = PerceptualLoss(device)

    def _cosine_beta_schedule(self, timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
        
    def noise_images(self, x, t):
        sqrt_alpha_bar = torch.sqrt(self.alpha_bars[t])[:, None, None, None]
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bars[t])[:, None, None, None]
        ε = torch.randn_like(x)
        return sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * ε, ε
    
    def _compute_single_scale_edge_loss(self, pred, target):
        def _get_edges(x):
            # Sobel kernels
            kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                  dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
            kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                  dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
            
            # Add padding
            pad = F.pad(x, (1, 1, 1, 1), mode='reflect')
            
            # Compute gradients
            grad_x = F.conv2d(pad, kernel_x)
            grad_y = F.conv2d(pad, kernel_y)
            
            return torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        
        pred_edges = _get_edges(pred)
        target_edges = _get_edges(target)
        
        return F.l1_loss(pred_edges, target_edges)
    
    def _compute_edge_loss(self, pred, target):
        """
        Compute multi-scale edge-aware loss between prediction and target.
        """
        scales = [1, 2, 4]
        total_edge_loss = self._compute_single_scale_edge_loss(pred, target)
        
        for scale in scales[1:]:  # Skip scale=1 as it's already computed
            pred_scaled = F.interpolate(pred, scale_factor=1/scale, mode='bilinear', align_corners=False)
            target_scaled = F.interpolate(target, scale_factor=1/scale, mode='bilinear', align_corners=False)
            edge_loss_scaled = self._compute_single_scale_edge_loss(pred_scaled, target_scaled)
            total_edge_loss += edge_loss_scaled
        
        return total_edge_loss / len(scales)
    
    def _sample_timesteps(self, batch_size):
        # Sample timesteps with emphasis on middle steps
        weights = torch.exp(-0.5 * ((torch.arange(self.n_steps, device=self.device) 
                                - self.n_steps//2) / (self.n_steps//4))**2)
        return torch.multinomial(weights, batch_size, replacement=True)

    def training_step(self, depth_maps, rgb=None, poses=None):
        batch_size = depth_maps.shape[0]
        
        # Sample timesteps
        t = self._sample_timesteps(batch_size)
        
        # Generate noisy depths
        noisy_depths, noise = self.noise_images(depth_maps, t)
        
        # Predict noise with conditioning
        noise_pred = self.model(noisy_depths, t/self.n_steps, rgb)
        
        # Multiple loss components
        base_loss = F.mse_loss(noise_pred, noise)
        edge_loss = self._compute_edge_loss(
            self._denoise(noisy_depths, noise_pred), 
            depth_maps
        )
        
        # Add multi-view consistency if poses available
        consistency_loss = (
            self._compute_consistency_loss(depth_maps, poses) 
            if poses is not None else 0
        )
        
        # Dynamic weighting
        total_loss = (
            base_loss + 
            self.config.edge_weight * edge_loss + 
            self.config.consistency_weight * consistency_loss
        )
        
        return total_loss
    
    def _denoise(self, noisy, pred_noise):
        """Helper method to denoise images given predicted noise"""
        return noisy - pred_noise
    
    def validation_step(self, depth_maps):
        batch_size = depth_maps.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), device=self.device)
        
        noisy_depths, noise = self.noise_images(depth_maps, t)
        noise_pred = self.model(noisy_depths, t/self.n_steps)
        
        return F.mse_loss(noise_pred, noise)


class PerceptualLoss(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        vgg = models.vgg16(pretrained=True).to(device)
        self.blocks = nn.ModuleList([
            vgg.features[:4],   # first conv block
            vgg.features[4:9],  # second conv block
            vgg.features[9:16], # third conv block
        ]).eval()
        
        for param in self.parameters():
            param.requires_grad = False
            
        self.transform = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, 3, 3, padding=0)  # Convert 1 channel to 3 channels
        ).to(device)
    
    def forward(self, x, y):
        x = self.transform(x)  # 1 -> 3 channels
        y = self.transform(y)  # 1 -> 3 channels
        
        x_features = []
        y_features = []
        
        for block in self.blocks:
            x = block(x)
            y = block(y)
            x_features.append(x)
            y_features.append(y)
        
        loss = 0.0
        for x_feat, y_feat in zip(x_features, y_features):
            loss += F.l1_loss(x_feat, y_feat)
            
        return loss / len(self.blocks)
    
# Add to diffusion.py
class MultiViewDiffusion(DiffusionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_extractor = FeatureExtractor()
        
    def _compute_consistency_loss(self, depth_maps, poses):
        """Compute consistency between multiple views"""
        batch_size = depth_maps.shape[0]
        features = []
        
        for i in range(batch_size):
            proj_depth = self._project_depth(depth_maps[i], poses[i])
            feat = self.feature_extractor(proj_depth)
            features.append(feat)
            
        consistency_loss = 0
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                consistency_loss += F.l1_loss(features[i], features[j])
                
        return consistency_loss / (batch_size * (batch_size - 1))
    
    def _project_depth(self, depth, pose):
        """Project depth map to new viewpoint"""
        batch_size = depth.shape[0]
        height, width = depth.shape[2:]
        
        # Create pixel coordinates
        y, x = torch.meshgrid(torch.arange(height), torch.arange(width))
        pixels = torch.stack([x, y, torch.ones_like(x)], dim=0).float()
        pixels = pixels.to(depth.device)
        
        # Unproject to 3D points
        points = depth.view(batch_size, 1, -1) * pixels.view(1, 3, -1)
        
        # Transform points
        R = pose[:, :3, :3]  # Rotation matrix
        t = pose[:, :3, 3:]  # Translation vector
        transformed_points = R @ points + t
        
        # Project to new depth
        new_depth = transformed_points[:, 2].view(batch_size, height, width)
        
        return new_depth.unsqueeze(1)