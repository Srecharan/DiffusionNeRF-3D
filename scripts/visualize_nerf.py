# scripts/visualize_nerf.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import os
import sys
from tqdm import tqdm
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.models.nerf import NeRF
from src.rendering.volumetric_rendering import VolumetricRenderer
from data.nerf_dataset import get_nerf_loaders

class NeRFVisualizer:
    def __init__(self, config, checkpoint_path):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.nerf_model = self._load_model(checkpoint_path)
        self.renderer = VolumetricRenderer(
            n_coarse=config['model']['nerf']['n_samples_coarse'],
            n_fine=config['model']['nerf']['n_samples_fine']
        )
        
        # Create output directory
        self.output_dir = Path('visualizations/nerf')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance monitoring
        self.perf_stats = {
            'fps': [],
            'memory_usage': [],
            'ray_batch_size': config['data']['nerf']['n_rays_per_batch']
        }
    
    def _load_model(self, checkpoint_path):
        model = NeRF(
            pos_encoding_dims=self.config['model']['nerf']['pos_encoding_dims'],
            view_encoding_dims=self.config['model']['nerf']['view_encoding_dims'],
            hidden_dims=self.config['model']['nerf']['hidden_dims']
        ).to(self.device)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model
    
    @torch.no_grad()
    def render_path(self, poses, image_size, chunk_size=32768):
        """Render images from a sequence of camera poses."""
        H, W = image_size
        rgbs = []
        depths = []
        
        for pose in tqdm(poses, desc='Rendering path'):
            # Generate rays for this pose
            rays_o, rays_d = self._get_rays(H, W, pose)
            rays_o = rays_o.to(self.device)
            rays_d = rays_d.to(self.device)
            
            # Render with chunks for memory efficiency
            rgb_chunks = []
            depth_chunks = []
            
            for i in range(0, rays_o.shape[0], chunk_size):
                chunk_rays_o = rays_o[i:i+chunk_size]
                chunk_rays_d = rays_d[i:i+chunk_size]
                
                with autocast(device_type='cuda', enabled=True):
                    out = self.renderer.render_rays(
                        self.nerf_model,
                        chunk_rays_o,
                        chunk_rays_d,
                        near=self.config['model']['nerf']['near'],
                        far=self.config['model']['nerf']['far']
                    )
                
                rgb_chunks.append(out['rgb_fine'].cpu())
                depth_chunks.append(out['depth_fine'].cpu())
            
            rgb = torch.cat(rgb_chunks, 0).reshape(H, W, 3)
            depth = torch.cat(depth_chunks, 0).reshape(H, W)
            
            rgbs.append(rgb)
            depths.append(depth)
        
        return torch.stack(rgbs), torch.stack(depths)
    
    def _get_rays(self, H, W, pose):
        """Generate rays for an image."""
        i, j = torch.meshgrid(torch.linspace(0, H-1, H),
                            torch.linspace(0, W-1, W))
        
        # Camera parameters (from NYU dataset)
        fx, fy = 582.62, 582.69
        cx, cy = 313.04, 238.44
        
        # Scale for image size
        fx *= W / 640
        fy *= H / 480
        cx *= W / 640
        cy *= H / 480
        
        dirs = torch.stack([(j-cx)/fx, -(i-cy)/fy, -torch.ones_like(i)], -1)
        rays_d = torch.sum(dirs[..., None, :] * pose[:3, :3], -1)
        rays_o = pose[:3, -1].expand(rays_d.shape)
        
        return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
    
    def generate_spiral_poses(self, n_frames=120):
        """Generate spiral path for video rendering."""
        poses = []
        for theta in np.linspace(0, 2*np.pi, n_frames):
            # Create spiral motion
            c2w = np.eye(4)
            c2w[0, 3] = np.cos(theta) * 0.2
            c2w[1, 3] = np.sin(theta) * 0.2
            c2w[2, 3] = -2 + np.sin(theta*2) * 0.1
            poses.append(torch.from_numpy(c2w.astype(np.float32)))
        return poses
    
    def visualize_results(self, val_loader):
        """Generate comprehensive visualization of results."""
        print("Generating visualizations...")
        
        # 1. Render validation images
        for idx, batch in enumerate(val_loader):
            if idx >= 5:  # Visualize first 5 images
                break
                
            rays_o = batch['rays_o'].to(self.device)
            rays_d = batch['rays_d'].to(self.device)
            rgb_gt = batch['rgb'].to(self.device)
            depth_gt = batch['depth'].to(self.device)
            
            # Render
            start_time = time.time()
            with autocast(device_type='cuda', enabled=True):
                out = self.renderer.render_rays(
                    self.nerf_model,
                    rays_o,
                    rays_d,
                    near=self.config['model']['nerf']['near'],
                    far=self.config['model']['nerf']['far']
                )
            elapsed = time.time() - start_time
            
            # Update performance stats
            self.perf_stats['fps'].append(1/elapsed)
            if torch.cuda.is_available():
                self.perf_stats['memory_usage'].append(
                    torch.cuda.memory_allocated() / 1024**2
                )
            
            # Create visualization
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # RGB comparison
            axes[0,0].imshow(rgb_gt[0].cpu())
            axes[0,0].set_title('Ground Truth RGB')
            
            axes[0,1].imshow(out['rgb_fine'][0].cpu())
            axes[0,1].set_title('Rendered RGB')
            
            diff_rgb = torch.abs(rgb_gt[0].cpu() - out['rgb_fine'][0].cpu())
            axes[0,2].imshow(diff_rgb)
            axes[0,2].set_title('RGB Error')
            
            # Depth comparison
            axes[1,0].imshow(depth_gt[0].cpu(), cmap='plasma')
            axes[1,0].set_title('Ground Truth Depth')
            
            axes[1,1].imshow(out['depth_fine'][0].cpu(), cmap='plasma')
            axes[1,1].set_title('Rendered Depth')
            
            diff_depth = torch.abs(depth_gt[0].cpu() - out['depth_fine'][0].cpu())
            im = axes[1,2].imshow(diff_depth, cmap='hot')
            axes[1,2].set_title('Depth Error')
            plt.colorbar(im, ax=axes[1,2])
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f'validation_{idx}.png', dpi=300)
            plt.close()
        
        # 2. Generate spiral path video
        poses = self.generate_spiral_poses()
        rgbs, depths = self.render_path(poses, 
                                      self.config['data']['nerf']['image_size'])
        
        # Save video frames
        video_dir = self.output_dir / 'video'
        video_dir.mkdir(exist_ok=True)
        
        for i, (rgb, depth) in enumerate(zip(rgbs, depths)):
            plt.figure(figsize=(20, 10))
            
            plt.subplot(121)
            plt.imshow(rgb)
            plt.title('RGB')
            
            plt.subplot(122)
            plt.imshow(depth, cmap='plasma')
            plt.title('Depth')
            plt.colorbar()
            
            plt.savefig(video_dir / f'frame_{i:04d}.png', dpi=150)
            plt.close()
        
        # 3. Plot performance metrics
        plt.figure(figsize=(15, 5))
        
        plt.subplot(121)
        plt.plot(self.perf_stats['fps'])
        plt.title('Rendering Speed (FPS)')
        plt.xlabel('Frame')
        plt.ylabel('FPS')
        
        if self.perf_stats['memory_usage']:
            plt.subplot(122)
            plt.plot(self.perf_stats['memory_usage'])
            plt.title('GPU Memory Usage')
            plt.xlabel('Frame')
            plt.ylabel('Memory (MB)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_metrics.png')
        plt.close()
        
        print(f"Visualizations saved to {self.output_dir}")
        print(f"Average FPS: {np.mean(self.perf_stats['fps']):.2f}")
        if self.perf_stats['memory_usage']:
            print(f"Peak GPU Memory: {max(self.perf_stats['memory_usage']):.2f} MB")

def main():
    # Load config
    with open('configs/model_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Get latest checkpoint
    latest_exp = max(Path('runs').glob('nerf_experiment_*'),
                    key=lambda p: p.stat().st_mtime)
    checkpoint_path = latest_exp / 'best.pt'
    
    # Create visualizer
    visualizer = NeRFVisualizer(config, checkpoint_path)
    
    # Get validation data
    _, val_loader = get_nerf_loaders(
        config['data']['dir'],
        batch_size=1,
        img_wh=tuple(config['data']['nerf']['image_size']),
        n_rays=config['data']['nerf']['n_rays_per_batch']
    )
    
    # Generate visualizations
    visualizer.visualize_results(val_loader)

if __name__ == '__main__':
    main()