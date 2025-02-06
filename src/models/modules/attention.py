import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim
        
        self.scale = dim_head ** -0.5
        self.heads = heads
        
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        
    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = dim // self.heads
        tensor = tensor.reshape(batch_size, seq_len, self.heads, head_size)
        tensor = tensor.permute(0, 2, 1, 3)
        return tensor.reshape(batch_size * self.heads, seq_len, head_size)

    def reshape_batch_dim_to_heads(self, tensor, batch_size):
        head_size = tensor.shape[-1]
        tensor = tensor.reshape(batch_size, self.heads, -1, head_size)
        tensor = tensor.permute(0, 2, 1, 3)
        return tensor.reshape(batch_size, -1, self.heads * head_size)
        
    def forward(self, x, context=None):
        batch_size, channel, height, width = x.shape
        
        x = x.reshape(batch_size, channel, -1).permute(0, 2, 1)
        if context is None:
            context = x
        
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        q = self.reshape_heads_to_batch_dim(q)
        k = self.reshape_heads_to_batch_dim(k)
        v = self.reshape_heads_to_batch_dim(v)
        
        # Attention
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        
        # Reshape back
        out = self.reshape_batch_dim_to_heads(out, batch_size)
        out = self.to_out(out)
        out = out.permute(0, 2, 1).reshape(batch_size, channel, height, width)
        return out

class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0.0, context_dim=None):
        super().__init__()
        self.attn1 = CrossAttention(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout
        )
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=dim, eps=1e-6)
        
        if context_dim is not None:
            self.attn2 = CrossAttention(
                query_dim=dim,
                context_dim=context_dim,
                heads=n_heads,
                dim_head=d_head,
                dropout=dropout
            )
            self.norm2 = nn.GroupNorm(num_groups=8, num_channels=dim, eps=1e-6)
        
        self.ff = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=dim, eps=1e-6),
            nn.Conv2d(dim, dim * 4, 1),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, 1)
        )
        
    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        if hasattr(self, 'attn2'):
            x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(x) + x
        return x