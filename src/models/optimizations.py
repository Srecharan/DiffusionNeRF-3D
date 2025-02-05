# src/models/optimizations.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

class OccupancyGrid:
    def __init__(self, resolution: int = 128, threshold: float = 0.01):
        """
        Initialize 3D occupancy grid for efficient ray sampling.
        
        Args:
            resolution: Grid resolution (same for all dimensions)
            threshold: Density threshold for considering a cell occupied
        """
        self.resolution = resolution
        self.threshold = threshold
        self.grid = torch.zeros((resolution,) * 3, dtype=torch.bool)
        self.updated = False
        
    def update_grid(self, nerf_model: nn.Module, device: torch.device):
        """Update occupancy grid using density predictions."""
        print("Updating occupancy grid...")
        
        # Create uniform grid points
        coords = torch.linspace(-1, 1, self.resolution)
        grid_points = torch.stack(torch.meshgrid(coords, coords, coords), -1)
        grid_points = grid_points.reshape(-1, 3).to(device)
        
        # Query density in batches
        density_grid = []
        with torch.no_grad():
            for i in range(0, len(grid_points), 64**3):
                points = grid_points[i:i+64**3]
                # Query model density without view dependence
                density = nerf_model(points, None, density_only=True)
                density_grid.append(density.cpu())
        
        density_grid = torch.cat(density_grid, 0)
        self.grid = (density_grid > self.threshold).reshape(self.resolution, 
                                                          self.resolution,
                                                          self.resolution)
        self.updated = True
        
    def sample_points(self, rays_o: torch.Tensor, rays_d: torch.Tensor, 
                     near: float, far: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample points along rays using occupancy grid.
        
        Args:
            rays_o: Ray origins (B, 3)
            rays_d: Ray directions (B, 3)
            near: Near plane distance
            far: Far plane distance
            
        Returns:
            Tuple of points (B, N, 3) and valid mask (B, N)
        """
        device = rays_o.device
        batch_size = rays_o.shape[0]
        
        # Convert rays to grid coordinates
        grid_size = 2.0  # Grid spans from -1 to 1
        cell_size = grid_size / self.resolution
        
        # Initialize samples
        t = torch.linspace(near, far, self.resolution).to(device)
        points = rays_o[..., None, :] + rays_d[..., None, :] * t[None, :, None]
        
        # Convert to grid indices
        grid_indices = ((points + 1) / cell_size).long()
        valid_mask = (grid_indices >= 0) & (grid_indices < self.resolution)
        valid_mask = valid_mask.all(-1)
        
        # Get occupancy values
        grid = self.grid.to(device)
        occupied = F.grid_sample(
            grid[None, None].float(),
            points.reshape(1, -1, 1, 3) / grid_size,
            align_corners=True
        ).reshape(batch_size, -1) > 0.5
        
        valid_mask = valid_mask & occupied
        
        return points, valid_mask

class HashEncoding(nn.Module):
    """
    Multi-resolution hash encoding for 3D coordinates.
    Implements the method described in Instant-NGP paper.
    
    Args:
        n_levels (int): Number of resolution levels for hierarchical hashing
        n_features (int): Number of features per level
        min_res (int): Minimum grid resolution (must be power of 2)
        max_res (int): Maximum grid resolution (must be power of 2)
        hash_size (int): Size of hash table (2^hash_size entries)
    """
    def __init__(self, n_levels=16, n_features=2, min_res=16, max_res=512, hash_size=19):
        super().__init__()
        self.n_levels = n_levels
        self.n_features = n_features
        self.hash_size = hash_size
        self.hash_table_size = 2**hash_size
        
        # Pre-compute and register resolutions for each level
        self.register_buffer('resolutions', torch.tensor([
            int(min_res * (max_res/min_res)**(i/(n_levels-1)))
            for i in range(n_levels)
        ]))

        # Pre-compute and register corner offsets for trilinear interpolation
        offsets = torch.stack(torch.meshgrid([
            torch.tensor([0, 1]) for _ in range(3)
        ], indexing='ij')).reshape(3, -1).t()
        self.register_buffer('offsets', offsets)
        
        # Create hash tables with optimized initialization
        self.hash_tables = nn.ModuleList([
            nn.Embedding(self.hash_table_size, n_features)
            for _ in range(n_levels)
        ])
        
        self._initialize_hash_tables()
    
    def _initialize_hash_tables(self):
        """Initialize hash table weights with small random values."""
        for table in self.hash_tables:
            # Initialize with small values for better training stability
            nn.init.uniform_(table.weight, -1e-4, 1e-4)
            
    def hash_fn(self, coords):
        """
        Compute spatial hash function for coordinates.
        
        Args:
            coords: tensor of shape [..., 3] containing coordinates in range [-1, 1]
            
        Returns:
            tensor of shape [...] containing hash values
        """
        # Scale coordinates to integers for hashing
        coords = ((coords + 1) * 0.5 * (2**18)).long()
        
        # Prime numbers for hashing
        primes = torch.tensor([1, 2654435761, 805459861], 
                            device=coords.device, dtype=torch.int64)
        
        # Compute hash using tensor operations
        hashed = (coords * primes[None, :]).sum(dim=-1)
        return hashed % self.hash_table_size

    def _interpolate_features(self, coords_floor, local_coords, features):
        """
        Perform trilinear interpolation of features at corner points.
        
        Args:
            coords_floor: integer coordinates of lower corner
            local_coords: fractional coordinates within grid cell
            features: feature vectors at corner points
            
        Returns:
            interpolated features
        """
        weights = torch.ones_like(local_coords[..., 0:1])
        for dim in range(3):
            w_dim = 1 - local_coords[..., dim:dim+1]
            weights = weights * torch.lerp(w_dim, 1-w_dim, self.offsets[:, dim].to(weights.device))
            
        return (weights * features).sum(dim=-2)

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x):
        """
        Forward pass: compute hash encoding for input coordinates.
        
        Args:
            x: tensor of shape [..., 3] containing coordinates in range [-1, 1]
            
        Returns:
            tensor of shape [..., n_levels * n_features] containing encoded features
        """
        # Ensure input is float32 and in valid range
        x = x.float()
        x = torch.clamp(x, -1, 1)
        
        outputs = []
        input_shape = x.shape[:-1]
        
        for i, resolution in enumerate(self.resolutions):
            # Scale coordinates to current resolution
            coords = x * resolution
            coords_floor = torch.floor(coords).long()
            local_coords = coords - coords_floor
            
            # Initialize feature accumulator
            interpolated = torch.zeros(
                (*coords.shape[:-1], self.n_features),
                device=coords.device,
                dtype=torch.float32
            )
            
            # Compute features for all corners
            corner_coords = (coords_floor[..., None, :] + 
                           self.offsets.to(coords.device)) % resolution
            corner_coords = corner_coords.float() / resolution  # Normalize to [-1, 1]
            
            # Hash and lookup features for all corners at once
            hashed = self.hash_fn(corner_coords)
            corner_features = self.hash_tables[i](hashed)
            
            # Interpolate features
            weights = torch.ones_like(local_coords[..., None, 0])
            for dim in range(3):
                w = local_coords[..., None, dim]
                weights = weights * torch.where(
                    self.offsets[:, dim].to(coords.device) == 0,
                    1 - w, w
                )
            
            # Compute final interpolated features
            interpolated = (weights * corner_features).sum(dim=-2)
            outputs.append(interpolated)
        
        # Concatenate features from all levels
        return torch.cat(outputs, dim=-1)

    def extra_repr(self):
        """Returns a string with extra representation information."""
        return f'n_levels={self.n_levels}, n_features={self.n_features}, hash_size={self.hash_size}'
    
    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x):
        """Forward pass with explicit type handling."""
        batch_size = x.shape[0]
        outputs = []
        
        # Ensure input is float32
        x = x.float()
        x = torch.clamp(x, -1, 1)
        
        for i, resolution in enumerate(self.resolutions):
            # Scale coordinates to current resolution
            coords = x * resolution
            coords_floor = torch.floor(coords).long()
            
            # Compute interpolation weights
            local_coords = coords - coords_floor.float()
            
            # Get corner coordinates
            corners = []
            weights = []
            
            for j in range(8):
                # Get corner
                corner = coords_floor + self.offsets[j].to(coords_floor.device)
                corner = corner % resolution  # Wrap around boundaries
                corners.append(corner)
                
                # Compute weight
                weight = torch.ones_like(local_coords[..., 0])
                for dim in range(3):
                    if self.offsets[j, dim] == 0:
                        weight = weight * (1 - local_coords[..., dim])
                    else:
                        weight = weight * local_coords[..., dim]
                weights.append(weight)
            
            corners = torch.stack(corners, dim=1)  # [batch_size, 8, 3]
            weights = torch.stack(weights, dim=1)  # [batch_size, 8]
            
            # Normalize coordinates for hashing
            corners = corners.float() / resolution  # Scale to [-1, 1]
            
            # Hash and lookup features
            hashed = self.hash_fn(corners)
            features = self.hash_tables[i](hashed)  # [batch_size, 8, n_features]
            
            # Interpolate features
            interpolated = torch.sum(weights.unsqueeze(-1) * features, dim=1)
            outputs.append(interpolated)
        
        return torch.cat(outputs, -1)

class NeRFOptimizer:
    def __init__(self, config: dict):
        """
        Handles NeRF performance optimizations.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.occupancy_grid = OccupancyGrid(
            resolution=config['model']['nerf']['occupancy_resolution']
        )
        self.hash_encoding = HashEncoding(
            n_levels=config['model']['nerf']['hash_num_levels'],
            n_features=config['model']['nerf']['hash_features_per_level'],
            hash_size=config['model']['nerf']['hash_size']
        )
        
    def update_occupancy(self, nerf_model: nn.Module, device: torch.device):
        """Update occupancy grid periodically."""
        if not self.occupancy_grid.updated:
            self.occupancy_grid.update_grid(nerf_model, device)
    
    def encode_points(self, points: torch.Tensor) -> torch.Tensor:
        """Encode points using hash encoding."""
        return self.hash_encoding(points)
    
    def sample_points(self, rays_o: torch.Tensor, rays_d: torch.Tensor,
                     near: float, far: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample points using occupancy grid."""
        return self.occupancy_grid.sample_points(rays_o, rays_d, near, far)