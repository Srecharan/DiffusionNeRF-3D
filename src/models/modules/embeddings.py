import torch
import torch.nn as nn
import math

class TimeEmbedding(nn.Module):
    def __init__(self, dim, scale=1.0):
        super().__init__()
        self.dim = dim
        self.scale = scale
        
    def forward(self, timesteps):
        """
        Create sinusoidal timestep embeddings.
        timesteps: (N,) tensor of integers from [0, max_period-1]
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        
        # Create embedding
        args = timesteps[:, None].float() * freqs[None] * self.scale
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        # Zero pad if dimension is odd
        if self.dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            
        return embedding

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            TimeEmbedding(frequency_embedding_size),
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, t):
        return self.mlp(t)