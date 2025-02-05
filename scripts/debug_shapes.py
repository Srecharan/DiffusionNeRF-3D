# scripts/debug_shapes.py

import torch
import sys
import os
import yaml

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.models.nerf import NeRF
from data.nerf_dataset import get_nerf_loaders
from src.rendering.volumetric_rendering import VolumetricRenderer, sample_pdf

def print_tensor_info(name, tensor):
    print(f"\n{name}:")
    print(f"Shape: {tensor.shape}")
    print(f"Type: {tensor.dtype}")
    print(f"Device: {tensor.device}")
    if tensor.numel() > 0:
        print(f"Value range: [{tensor.min().item():.4f}, {tensor.max().item():.4f}]")

def debug_pdf_sampling(z_vals_mid, weights_c, n_fine=128):
    """Debug the PDF sampling process specifically."""
    print("\nPDF Sampling Debug:")
    print("-" * 50)
    print_tensor_info("z_vals_mid", z_vals_mid)
    print_tensor_info("weights_c", weights_c)
    
    # Get normalized weights (PDF)
    pdf = weights_c / torch.sum(weights_c, dim=-1, keepdim=True)
    print_tensor_info("pdf", pdf)
    
    # Get CDF
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)
    print_tensor_info("cdf", cdf)
    
    # Generate sample points
    u = torch.rand(list(cdf.shape[:-1]) + [n_fine], device=weights_c.device)
    print_tensor_info("u", u)
    
    # Find indices
    inds = torch.searchsorted(cdf, u)
    print_tensor_info("searchsorted indices", inds)
    below = torch.clamp(inds-1, min=0)
    above = torch.clamp(inds, max=cdf.shape[-1]-1)
    inds_g = torch.stack([below, above], -1)
    print_tensor_info("final indices", inds_g)
    
    return {
        'cdf_shape': cdf.shape,
        'inds_g_shape': inds_g.shape,
        'u_shape': u.shape
    }

def debug_shapes():
    # Load config
    with open('configs/model_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create data loader
    train_loader, _ = get_nerf_loaders(
        config['data']['dir'],
        batch_size=1,
        img_wh=(256, 256),
        n_rays=4096
    )
    
    # Get a batch
    batch = next(iter(train_loader))
    print("\nBatch contents:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print_tensor_info(k, v)
    
    # Initialize model and renderer
    model = NeRF(
        pos_encoding_dims=config['model']['nerf']['pos_encoding_dims'],
        view_encoding_dims=config['model']['nerf']['view_encoding_dims'],
        hidden_dims=config['model']['nerf']['hidden_dims'],
        use_hash_encoding=True
    ).to(device)
    
    renderer = VolumetricRenderer(
        n_coarse=config['model']['nerf']['n_samples_coarse'],
        n_fine=config['model']['nerf']['n_samples_fine']
    )
    
    # Move batch to device and run initial sampling
    rays_o = batch['rays_o'].to(device)
    rays_d = batch['rays_d'].to(device)
    
    # Get initial samples
    z_vals = renderer._stratified_sampling(0.1, 10.0, rays_o.shape[0], rays_o.shape[1], 
                                         renderer.n_coarse, device)
    print_tensor_info("z_vals", z_vals)
    
    # Process the first part of ray rendering
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    dirs = rays_d[..., None, :].expand_as(pts)
    pts_flat = pts.reshape(-1, 3)
    dirs_flat = dirs.reshape(-1, 3)
    raw_rgb, raw_density = model(pts_flat, dirs_flat)
    
    # Reshape outputs
    raw_rgb = raw_rgb.reshape(rays_o.shape[0], rays_o.shape[1], renderer.n_coarse, 3)
    raw_density = raw_density.reshape(rays_o.shape[0], rays_o.shape[1], renderer.n_coarse, 1)
    
    # Get weights
    weights = renderer._compute_weights(raw_density, z_vals, rays_d, device)
    print_tensor_info("weights from compute_weights", weights)
    
    # Debug fine sampling
    z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    weights_c = weights.squeeze(-1)[..., :-1]
    
    # Run detailed PDF sampling debug
    pdf_debug_info = debug_pdf_sampling(z_vals_mid, weights_c, renderer.n_fine)
    print("\nPDF Sampling Shapes:")
    for k, v in pdf_debug_info.items():
        print(f"{k}: {v}")

if __name__ == '__main__':
    debug_shapes()