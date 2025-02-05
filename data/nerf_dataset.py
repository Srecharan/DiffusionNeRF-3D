# src/data/nerf_dataset.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from scipy.spatial.transform import Rotation

class NeRFNYUDataset(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(256, 256), 
                 n_rays=1024, precrop_fraction=0.5):
        """
        Args:
            root_dir: NYU dataset root directory
            split: 'train' or 'val'
            img_wh: image width and height
            n_rays: number of rays to sample per image
            precrop_fraction: fraction of image to sample rays from during early training
        """
        self.root_dir = root_dir
        self.split = split
        self.img_wh = img_wh
        self.n_rays = n_rays
        self.precrop_fraction = precrop_fraction
        
        # Get file paths
        self.rgb_dir = os.path.join(root_dir, 'rgb')
        self.depth_dir = os.path.join(root_dir, 'depth')
        self.files = sorted(os.listdir(self.rgb_dir))
        
        # Split dataset
        split_idx = int(len(self.files) * 0.8)
        self.files = self.files[:split_idx] if split == 'train' else self.files[split_idx:]
        
        # Camera parameters (NYU v2 specific)
        self.K = np.array([[582.62, 0, 313.04],
                          [0, 582.69, 238.44],
                          [0, 0, 1]])
        
        # Scale camera matrix for resized images
        self.K[0] *= img_wh[0] / 640
        self.K[1] *= img_wh[1] / 480
        
        self.directions = self.get_ray_directions()
        self.epoch = 0  # For curriculum learning
        
    def get_ray_directions(self):
        """Calculate ray directions for all pixels."""
        i, j = torch.meshgrid(
            torch.linspace(0, self.img_wh[1]-1, self.img_wh[1]),
            torch.linspace(0, self.img_wh[0]-1, self.img_wh[0]),
            indexing='ij'  # Important: specify indexing
        )
        
        directions = torch.stack([
            (j - self.K[0,2]) / self.K[0,0],
            -(i - self.K[1,2]) / self.K[1,1],
            -torch.ones_like(i)
        ], -1)
        
        # Normalize directions
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)
        
        return directions

    def estimate_pose_from_depth(self, depth):
        """Estimate camera pose from depth using PnP."""
        # Sample 3D points from depth
        h, w = depth.shape
        i, j = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        points_2d = np.stack([j, i], axis=-1).reshape(-1, 2).astype(np.float32)
        
        # Convert to 3D points
        z = depth.reshape(-1)
        x = (points_2d[:, 0] - self.K[0,2]) * z / self.K[0,0]
        y = (points_2d[:, 1] - self.K[1,2]) * z / self.K[1,1]
        points_3d = np.stack([x, y, z], axis=-1)
        
        # Filter valid points (non-zero depth and within reasonable range)
        valid_mask = (z > 0.1) & (z < 10.0)
        points_3d = points_3d[valid_mask]
        points_2d = points_2d[valid_mask]
        
        # Ensure we have enough points
        min_points = 100
        if len(points_3d) < min_points:
            return np.eye(4)
        
        # Randomly sample points if we have too many
        if len(points_3d) > 1000:
            indices = np.random.choice(len(points_3d), 1000, replace=False)
            points_3d = points_3d[indices]
            points_2d = points_2d[indices]
        
        try:
            # Estimate pose using PnP
            _, R_exp, t = cv2.solvePnP(
                points_3d.astype(np.float32),
                points_2d.astype(np.float32),
                self.K.astype(np.float32),
                None,
                flags=cv2.SOLVEPNP_EPNP
            )
            
            R, _ = cv2.Rodrigues(R_exp)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t.squeeze()
            
            return T
        except cv2.error:
            print("PnP failed, returning identity pose")
            return np.eye(4)
    
    def get_rays(self, c2w):
        """Get ray origins and directions in world coordinate system."""
        # Ensure c2w is the right shape
        if c2w.dim() == 2:
            c2w = c2w.unsqueeze(0)  # Add batch dimension if needed
        
        # Get directions in camera space
        directions = self.get_ray_directions()  # (H, W, 3)
        
        # Transform ray directions to world space
        rays_d = torch.sum(directions[..., None, :] * c2w[..., :3, :3], -1)
        
        # Get ray origins (camera position in world space)
        rays_o = c2w[..., :3, -1].expand(rays_d.shape)
        
        return rays_o, rays_d
    
    def sample_rays(self, rays_o, rays_d, rgb, depth, precrop=True):
        """Sample rays during training."""
        if self.split == 'train':
            if precrop and self.epoch < 500:  # Curriculum learning
                dH = int(self.img_wh[1] * self.precrop_fraction)
                dW = int(self.img_wh[0] * self.precrop_fraction)
                
                # Calculate center region
                start_h = self.img_wh[1]//2 - dH//2
                start_w = self.img_wh[0]//2 - dW//2
                
                # Create coordinate grid for the cropped region
                i, j = torch.meshgrid(
                    torch.arange(start_h, start_h + dH),
                    torch.arange(start_w, start_w + dW)
                )
            else:
                # Full image coordinates
                i, j = torch.meshgrid(
                    torch.arange(self.img_wh[1]),
                    torch.arange(self.img_wh[0])
                )
            
            # Flatten coordinates
            coords = torch.stack([i.flatten(), j.flatten()], -1)  # (H*W, 2)
            
            # Randomly select rays
            select_inds = np.random.choice(coords.shape[0], size=[self.n_rays], replace=False)
            select_coords = coords[select_inds]
            
            # Extract selected rays
            rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]
            rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]
            rgb = rgb[select_coords[:, 0], select_coords[:, 1]]
            depth = depth[select_coords[:, 0], select_coords[:, 1]]
            
            # Ensure all tensors have the correct shape
            rays_o = rays_o.reshape(self.n_rays, -1)
            rays_d = rays_d.reshape(self.n_rays, -1)
            rgb = rgb.reshape(self.n_rays, -1) if rgb.dim() == 1 else rgb.reshape(self.n_rays, 3)
            depth = depth.reshape(self.n_rays)
            
        return rays_o, rays_d, rgb, depth
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        rgb_path = os.path.join(self.rgb_dir, self.files[idx])
        depth_path = os.path.join(self.depth_dir, self.files[idx])
        
        # Load RGB and depth
        rgb = np.load(rgb_path).astype(np.float32) / 255.0
        depth = np.load(depth_path).astype(np.float32)
        
        # Resize images
        rgb = cv2.resize(rgb, self.img_wh[::-1])  # Note: cv2.resize expects (width, height)
        depth = cv2.resize(depth, self.img_wh[::-1])
        
        # Convert to torch tensors
        rgb = torch.from_numpy(rgb)
        depth = torch.from_numpy(depth)
        
        # Handle NaN and inf values in depth
        depth = torch.nan_to_num(depth, nan=0.0, posinf=10.0, neginf=0.0)
        depth = torch.clamp(depth, min=0.1, max=10.0)
        
        # Estimate camera pose
        c2w = self.estimate_pose_from_depth(depth.numpy())
        c2w = torch.from_numpy(c2w.astype(np.float32))
        
        # Get rays
        rays_o, rays_d = self.get_rays(c2w)
        
        # Sample rays during training
        if self.split == 'train':
            rays_o, rays_d, rgb, depth = self.sample_rays(rays_o, rays_d, rgb, depth)
        
        return {
            'rays_o': rays_o,
            'rays_d': rays_d,
            'rgb': rgb,
            'depth': depth,
            'c2w': c2w
        }

def get_nerf_loaders(root_dir, batch_size=1, img_wh=(256, 256), n_rays=1024):
    """Create data loaders for NeRF training."""
    train_dataset = NeRFNYUDataset(root_dir, split='train', img_wh=img_wh, n_rays=n_rays)
    val_dataset = NeRFNYUDataset(root_dir, split='val', img_wh=img_wh, n_rays=n_rays)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,  # Validate one image at a time
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader