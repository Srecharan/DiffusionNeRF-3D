# src/models/nerf.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .optimizations import HashEncoding
from torch.utils.checkpoint import checkpoint

class NeRF(nn.Module):
    def __init__(self, 
                 pos_encoding_dims=10,
                 view_encoding_dims=4,
                 hidden_dims=256,
                 num_layers=8,
                 skip_connections=(4,),
                 use_hash_encoding=False):
        super().__init__()
        self.use_checkpoint = True  # Use checkpointing for memory efficiency
        self.use_hash_encoding = use_hash_encoding
        self.pos_encoding_dims = pos_encoding_dims
        self.view_encoding_dims = view_encoding_dims
        
        # Initialize hash encoding if enabled
        if use_hash_encoding:
            self.hash_encoder = HashEncoding(
                n_levels=16,  # Increased levels for better detail
                n_features=2,
                min_res=16,
                max_res=1024,  # Increased max resolution
                hash_size=19  # Increased hash size
            )
            pos_enc_dims = 16 * 2  # Match n_levels
        else:
            pos_enc_dims = 3 + 3 * 2 * pos_encoding_dims
            
        view_enc_dims = 3 + 3 * 2 * view_encoding_dims
        
        # Layers for processing position with larger capacity
        self.pos_layers = nn.ModuleList()
        prev_dim = pos_enc_dims
        
        for i in range(num_layers):
            if i == 0:
                layer = nn.Linear(prev_dim, hidden_dims)
            elif i in skip_connections:
                layer = nn.Linear(hidden_dims + pos_enc_dims, hidden_dims)
            else:
                layer = nn.Linear(hidden_dims, hidden_dims)
            
            # Initialize weights for better training
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            
            self.pos_layers.append(layer)
            prev_dim = hidden_dims
        
        # Enhanced density head
        self.density_head = nn.Sequential(
            nn.Linear(hidden_dims, hidden_dims//2),
            nn.ReLU(),
            nn.Linear(hidden_dims//2, 1),
            nn.ReLU()
        )
        
        # Enhanced feature head
        self.feature_head = nn.Sequential(
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU()
        )
        
        # Enhanced view-dependent color layers
        self.view_layers = nn.Sequential(
            nn.Linear(hidden_dims + view_enc_dims, hidden_dims//2),
            nn.ReLU(),
            nn.Linear(hidden_dims//2, hidden_dims//4),
            nn.ReLU(),
            nn.Linear(hidden_dims//4, 3),
            nn.Sigmoid()
        )
        
        # Initialize remaining weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def positional_encoding(self, x, num_dims):
        """Apply positional encoding to input."""
        if num_dims == 0:
            return x
            
        encodings = [x]
        freq_bands = 2.0 ** torch.linspace(0., num_dims-1, num_dims, device=x.device)
        
        for freq in freq_bands:
            encodings.append(torch.sin(freq * x))
            encodings.append(torch.cos(freq * x))
                
        return torch.cat(encodings, dim=-1)
    
    def forward(self, x, view_dirs, density_only=False):
        """Forward pass through the network."""
        if self.training:
            # Regular forward during training
            return self._forward(x, view_dirs, density_only)
        else:
            # No checkpointing during validation/inference
            with torch.no_grad():
                return self._forward(x, view_dirs, density_only)

    def _forward(self, x, view_dirs, density_only=False):
        """Forward pass implementation."""
        if self.use_hash_encoding:
            h = self.hash_encoder(x)
        else:
            h = self.positional_encoding(x, self.pos_encoding_dims)
        
        # Process position features with skip connections
        input_h = h
        for i, layer in enumerate(self.pos_layers):
            if i in (4,):  # Skip connection
                h = torch.cat([h, input_h], dim=-1)
            h = F.relu(layer(h))
        
        # Compute density
        density = self.density_head(h)
        
        if density_only:
            return density
        
        # Compute view-dependent color
        features = self.feature_head(h)
        
        if view_dirs is not None:
            view_encoded = self.positional_encoding(view_dirs, self.view_encoding_dims)
            view_input = torch.cat([features, view_encoded], dim=-1)
            rgb = self.view_layers(view_input)
        else:
            rgb = torch.zeros_like(x).to(x.device)
        
        return rgb, density