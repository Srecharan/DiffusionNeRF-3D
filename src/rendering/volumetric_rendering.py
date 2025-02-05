# src/rendering/volumetric_rendering.py

import torch
import torch.nn.functional as F

def sample_pdf(bins, weights, n_samples, det=False):
    """Sample points from probability distribution given by weights.
    
    This is NOT for creating PDF files - it's for sampling points according to their importance
    during ray marching. PDF here means "Probability Density Function" which helps focus
    samples where they're most needed.
    
    Args:
        bins: tensor of shape [..., n_bins] (e.g., [1, 4096, 63])
        weights: tensor of shape [..., n_bins] (e.g., [1, 4096, 63])
        n_samples: number of samples to draw (e.g., 128)
        det: deterministic sampling
    """
    # Ensure float32 type
    weights = weights.float()
    bins = bins.float()
    
    # Add small epsilon to prevent NaNs
    weights = weights + 1e-5
    
    # Normalize weights to get PDF
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    
    # Get CDF
    cdf = torch.cumsum(pdf, dim=-1)  # [..., n_bins]
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)  # [..., n_bins+1]
    
    # Take sample positions
    if det:
        u = torch.linspace(0., 1., n_samples, device=weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples], device=weights.device)
    
    # Invert CDF to get samples
    u = u.contiguous()
    
    # Expand CDF for broadcasting
    cdf = cdf.unsqueeze(-2).expand(*cdf.shape[:-1], n_samples, cdf.shape[-1])
    
    # Find indices of surrounding CDF values
    inds = torch.searchsorted(cdf[..., -1], u)
    below = torch.clamp(inds-1, min=0)
    above = torch.clamp(inds, max=cdf.shape[-1]-1)
    inds_g = torch.stack([below, above], -1)
    
    # Expand bins for gathering
    bins_expanded = bins.unsqueeze(-2).expand(*bins.shape[:-1], n_samples, bins.shape[-1])
    
    # Get surrounding bin values
    bins_g = torch.gather(bins_expanded, -1, inds_g)
    cdf_g = torch.gather(cdf, -1, inds_g)
    
    # Linear interpolation
    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
    
    return samples

class VolumetricRenderer:
    def __init__(self, n_coarse=64, n_fine=128):
        self.n_coarse = n_coarse
        self.n_fine = n_fine
        self.training = True

    def _stratified_sampling(self, near, far, batch_size, n_rays, n_samples, device):
        """Generate stratified sample points along rays."""
        # Handle near/far as either scalars or tensors
        if isinstance(near, (int, float)):
            near = torch.tensor([near], device=device).expand(batch_size, n_rays)
        elif isinstance(near, torch.Tensor):
            if near.dim() == 0:
                near = near.expand(batch_size, n_rays)
            elif near.dim() == 1:
                near = near.unsqueeze(0).expand(batch_size, n_rays)
                
        if isinstance(far, (int, float)):
            far = torch.tensor([far], device=device).expand(batch_size, n_rays)
        elif isinstance(far, torch.Tensor):
            if far.dim() == 0:
                far = far.expand(batch_size, n_rays)
            elif far.dim() == 1:
                far = far.unsqueeze(0).expand(batch_size, n_rays)
        
        # Ensure near and far have the correct device
        near = near.to(device)
        far = far.to(device)
        
        # Create sampling points
        t_vals = torch.linspace(0., 1., n_samples, device=device)
        
        # Expand t_vals to match batch dimensions [batch_size, n_rays, n_samples]
        t_vals = t_vals.expand(batch_size, n_rays, n_samples)
        
        # Compute z values
        z_vals = near[..., None] * (1.-t_vals) + far[..., None] * t_vals
        
        if self.training:
            # Get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            
            # Stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape, device=device)
            z_vals = lower + (upper - lower) * t_rand
        
        return z_vals

    def _compute_weights(self, raw_density, z_vals, rays_d, device):
        """Helper function to compute weights from density values with improved numerical stability."""
        # Ensure correct shapes and types
        raw_density = raw_density.to(device, dtype=torch.float32)
        z_vals = z_vals.to(device, dtype=torch.float32)
        rays_d = rays_d.to(device, dtype=torch.float32)
        
        if raw_density.dim() == 3:
            raw_density = raw_density.unsqueeze(0)
            
        # Compute distances between samples
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e4)], -1)
        
        # Normalize ray directions and compute real world distances
        rays_d_norm = torch.norm(rays_d[..., None, :], dim=-1, keepdim=True).clamp(min=1e-8)
        dists = dists * rays_d_norm
        dists = dists[..., None]  # Add channel dimension
        
        # Clamp density values for stability
        density = F.relu(raw_density).clamp(min=0.0, max=1e8)
        
        # Compute alpha values with numerical stability
        alpha = 1.0 - torch.exp(-density * dists.clamp(min=1e-10))
        alpha = alpha.view_as(dists)
        
        # Ensure alpha is in valid range
        alpha = alpha.clamp(min=0.0, max=1.0)
        
        # Compute transmittance with improved stability
        transparency = (1.0 - alpha + 1e-10).clamp(min=1e-10, max=1.0)
        transmittance = torch.cat([
            torch.ones_like(alpha[..., :1, :]),
            torch.cumprod(transparency, dim=-2)[..., :-1, :]
        ], dim=-2)
        
        # Compute and check weights
        weights = alpha * transmittance
        weights = weights / (weights.sum(dim=-2, keepdim=True) + 1e-10)

    def render_rays(self, model, rays_o, rays_d, near, far, depth_prior=None):
        """Volumetric rendering for a batch of rays."""
        with torch.set_grad_enabled(self.training):
            device = rays_o.device
            batch_size, n_rays = rays_o.shape[:2]
            
            # Split into chunks for memory efficiency
            chunk_size = 1024
            outputs = []
            
            for i in range(0, n_rays, chunk_size):
                chunk_rays_o = rays_o[:, i:i+chunk_size]
                chunk_rays_d = rays_d[:, i:i+chunk_size]
                chunk_out = self._render_rays_chunk(
                    model, chunk_rays_o, chunk_rays_d, near, far, depth_prior
                )
                outputs.append(chunk_out)
            
            # Combine chunks
            combined = {}
            for k in outputs[0].keys():
                combined[k] = torch.cat([out[k] for out in outputs], dim=1)
            
            return combined
    
    def _render_rays_chunk(self, model, rays_o, rays_d, near, far, depth_prior=None):
        device = rays_o.device
        batch_size, n_rays = rays_o.shape[:2]
        
        # Sample points along each ray for coarse network
        z_vals = self._stratified_sampling(near, far, batch_size, n_rays, self.n_coarse, device)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        dirs = rays_d[..., None, :].expand_as(pts)
        
        # Flatten points and directions for processing
        pts_flat = pts.reshape(-1, 3)
        dirs_flat = dirs.reshape(-1, 3)
        
        # Process in smaller chunks for memory efficiency
        sub_chunk_size = min(8192, pts_flat.shape[0])
        raw_rgb_chunks = []
        raw_density_chunks = []
        
        for i in range(0, pts_flat.shape[0], sub_chunk_size):
            end_idx = min(i + sub_chunk_size, pts_flat.shape[0])
            pts_chunk = pts_flat[i:end_idx]
            dirs_chunk = dirs_flat[i:end_idx]
            
            rgb_chunk, density_chunk = model(pts_chunk, dirs_chunk)
            raw_rgb_chunks.append(rgb_chunk)
            raw_density_chunks.append(density_chunk)
        
        # Concatenate results
        raw_rgb = torch.cat(raw_rgb_chunks, 0)
        raw_density = torch.cat(raw_density_chunks, 0)
        
        # Reshape to correct dimensions
        raw_rgb = raw_rgb.reshape(batch_size, n_rays, self.n_coarse, 3)
        raw_density = raw_density.reshape(batch_size, n_rays, self.n_coarse, 1)
        
        # Compute weights for coarse samples
        weights_coarse = self._compute_weights(raw_density, z_vals, rays_d, device)
        rgb_coarse = torch.sum(weights_coarse * raw_rgb, dim=-2)
        depth_coarse = torch.sum(weights_coarse.squeeze(-1) * z_vals, dim=-1)
        
        # Sample fine points
        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        weights_coarse = weights_coarse.squeeze(-1)[..., :-1]
        
        z_samples = sample_pdf(
            z_vals_mid,
            weights_coarse,
            self.n_fine,
            det=(not self.training)
        )
        
        # Combine samples
        z_vals_combined, _ = torch.sort(torch.cat([z_vals, z_samples], dim=-1), dim=-1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals_combined[..., :, None]
        dirs = rays_d[..., None, :].expand_as(pts)
        
        # Process fine samples
        pts_flat = pts.reshape(-1, 3)
        dirs_flat = dirs.reshape(-1, 3)
        
        raw_rgb_chunks = []
        raw_density_chunks = []
        
        for i in range(0, pts_flat.shape[0], sub_chunk_size):
            end_idx = min(i + sub_chunk_size, pts_flat.shape[0])
            pts_chunk = pts_flat[i:end_idx]
            dirs_chunk = dirs_flat[i:end_idx]
            
            rgb_chunk, density_chunk = model(pts_chunk, dirs_chunk)
            raw_rgb_chunks.append(rgb_chunk)
            raw_density_chunks.append(density_chunk)
        
        raw_rgb = torch.cat(raw_rgb_chunks, 0)
        raw_density = torch.cat(raw_density_chunks, 0)
        
        n_total_samples = self.n_coarse + self.n_fine
        raw_rgb = raw_rgb.reshape(batch_size, n_rays, n_total_samples, 3)
        raw_density = raw_density.reshape(batch_size, n_rays, n_total_samples, 1)
        
        # Compute weights for fine samples
        weights = self._compute_weights(raw_density, z_vals_combined, rays_d, device)
        
        # Composite fine samples
        rgb_fine = torch.sum(weights * raw_rgb, dim=-2)
        depth_fine = torch.sum(weights.squeeze(-1) * z_vals_combined, dim=-1)
        
        return {
            'rgb_coarse': rgb_coarse,
            'rgb_fine': rgb_fine,
            'depth_coarse': depth_coarse,
            'depth_fine': depth_fine,
            'weights': weights
        }