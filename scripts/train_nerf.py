# scripts/train_nerf.py

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import yaml
import os
import sys
import numpy as np
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.models.nerf import NeRF
from src.models.unet import UNet
from src.models.diffusion import DiffusionModel
from src.models.optimizations import NeRFOptimizer
from data.nerf_dataset import get_nerf_loaders
from src.rendering.volumetric_rendering import VolumetricRenderer
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'

class NeRFTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.global_step = 0
        
        self.nerf_model = NeRF(
            pos_encoding_dims=config['model']['nerf']['pos_encoding_dims'],
            view_encoding_dims=config['model']['nerf']['view_encoding_dims'],
            hidden_dims=config['model']['nerf']['hidden_dims'],
            num_layers=config['model']['nerf']['num_layers'],
            skip_connections=config['model']['nerf']['skip_connections'],
            use_hash_encoding=config['model']['nerf']['use_hash_encoding']
        ).to(self.device)

        for param in self.nerf_model.parameters():
            param.requires_grad = True
        
        self.diffusion_model = self._load_diffusion_model()

        self.renderer = VolumetricRenderer(
            n_coarse=config['model']['nerf']['n_samples_coarse'],
            n_fine=config['model']['nerf']['n_samples_fine']
        )

        self.nerf_optimizer = NeRFOptimizer(config)

        self.optimizer = torch.optim.Adam([
            {'params': [p for n, p in self.nerf_model.named_parameters() if 'encoding' not in n], 
             'lr': config['training']['learning_rate']},
            {'params': [p for n, p in self.nerf_model.named_parameters() if 'encoding' in n], 
             'lr': config['training']['learning_rate'] * 10}  # Higher LR for encoding
        ])

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[100000, 150000],
            gamma=0.1
        )

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = os.path.join('runs', f'nerf_experiment_{timestamp}')
        self.writer = SummaryWriter(self.log_dir)
 
        os.makedirs(self.log_dir, exist_ok=True)
        with open(os.path.join(self.log_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config, f)
    
    def _load_diffusion_model(self):
        """Load pre-trained diffusion model."""
        unet = UNet(
            in_channels=1,
            out_channels=1,
            time_emb_dim=512,  
            base_channels=128,  
            attention=True
        ).to(self.device)
        
        diffusion = DiffusionModel(
            unet,
            n_steps=self.config['diffusion']['n_steps'],
            beta_schedule=self.config['diffusion']['beta_schedule'],
            beta_start=self.config['diffusion']['beta_start'],
            beta_end=self.config['diffusion']['beta_end'],
            device=self.device
        )

        experiment_dir = 'diffusion_experiment_20250113_211656'  # Your best model
        checkpoint_path = os.path.join('runs', experiment_dir, 'best_model.pt')
        
        print(f"Loading diffusion model from: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        
        for key, value in state_dict.items():
            if key.startswith('_orig_mod.'):
                new_key = key.replace('_orig_mod.', '')
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value

        unet.load_state_dict(new_state_dict)
        unet.eval()
        
        return diffusion
    
    def _compute_depth_loss(self, pred_depth, gt_depth):
        """Compute depth loss with robust edge-awareness and proper handling of invalid depths."""
        if pred_depth.dim() == 1:
            pred_depth = pred_depth.unsqueeze(0)
        if gt_depth.dim() == 1:
            gt_depth = gt_depth.unsqueeze(0)

        valid_mask = (gt_depth > 0).float()
        
        # L1 loss only on valid regions
        l1_loss = F.l1_loss(pred_depth * valid_mask, gt_depth * valid_mask, reduction='sum') / (valid_mask.sum() + 1e-8)
        
        try:
            edge_loss = self._compute_edge_loss(pred_depth, gt_depth)
        except Exception as e:
            print(f"Warning: Edge loss computation failed, falling back to zero")
            edge_loss = torch.tensor(0.0, device=pred_depth.device, requires_grad=True)
        
        alpha = min(self.global_step / 100000, 0.5) 
        
        diff = torch.abs(pred_depth - gt_depth) * valid_mask
        delta = 0.2 * torch.max(diff).item()
        berhu_loss = torch.where(diff <= delta,
                            diff,
                            (diff * diff + delta * delta) / (2 * delta))
        berhu_loss = torch.sum(berhu_loss) / (valid_mask.sum() + 1e-8)
        
        total_loss = l1_loss + alpha * edge_loss + 0.5 * berhu_loss

        return total_loss

    def _compute_edge_loss(self, pred, target):
        """Compute edge-aware loss between prediction and target with robust grid handling."""
        def _get_edges(x):
            """Get edges using Sobel filters with robust grid size handling."""
            if x.dim() == 1:
                x = x.unsqueeze(0)

            batch_size = x.shape[0]
            n_rays = x.shape[1] if x.dim() > 1 else x.size(0)
            
            grid_size = int(np.ceil(np.sqrt(n_rays)))
            total_size = grid_size * grid_size
            
            padding_size = total_size - n_rays
            if padding_size > 0:
                padding = torch.zeros(batch_size, padding_size, device=x.device)
                x = torch.cat([x, padding], dim=-1)
            
            try:
                x = x.view(batch_size, 1, grid_size, grid_size)
                
                sobel_x = torch.tensor([[-1, 0, 1], 
                                    [-2, 0, 2], 
                                    [-1, 0, 1]], device=x.device).float()
                sobel_y = torch.tensor([[-1, -2, -1],
                                    [0, 0, 0],
                                    [1, 2, 1]], device=x.device).float()
                
                sobel_x = sobel_x.view(1, 1, 3, 3)
                sobel_y = sobel_y.view(1, 1, 3, 3)

                x = F.pad(x, (1, 1, 1, 1), mode='reflect')

                edges_x = F.conv2d(x, sobel_x)
                edges_y = F.conv2d(x, sobel_y)
                edges = torch.sqrt(edges_x.pow(2) + edges_y.pow(2) + 1e-6)
                edges = edges.view(batch_size, -1)[:, :n_rays]
                
                return edges
            
            except Exception as e:
                print(f"Error in edge detection - shape: {x.shape}, grid_size: {grid_size}")
                print(f"Original n_rays: {n_rays}, padded size: {total_size}")
                raise e
        
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)
        if target.dim() == 1:
            target = target.unsqueeze(0)
        
        valid_mask = (target > 0).float()
        pred = pred * valid_mask
        target = target * valid_mask
        
        try:
            pred_edges = _get_edges(pred)
            target_edges = _get_edges(target)
            
            # Compute loss
            edge_loss = F.l1_loss(pred_edges, target_edges)
            return edge_loss
        except RuntimeError as e:
            print(f"Warning: Edge loss computation failed, using zero loss")
            print(f"Error: {str(e)}")
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        
    def train_step(self, batch):
        try:
            self.nerf_model.train()
            self.renderer.training = True
            self.optimizer.zero_grad()
    
            rays_o = batch['rays_o'].to(self.device)
            rays_d = batch['rays_d'].to(self.device)
            rgb_gt = batch['rgb'].to(self.device)
            depth_gt = batch['depth'].to(self.device)
            assert rays_o.isfinite().all(), "rays_o contains inf/nan"
            assert rays_d.isfinite().all(), "rays_d contains inf/nan"
            
            with torch.set_grad_enabled(True):
                render_out = self.renderer.render_rays(
                    self.nerf_model,
                    rays_o,
                    rays_d,
                    near=0.1,
                    far=10.0,
                    depth_prior=depth_gt
                )
                
                assert render_out['rgb_fine'].isfinite().all(), "RGB output contains inf/nan"
                assert render_out['depth_fine'].isfinite().all(), "Depth output contains inf/nan"
                
                rgb_loss = F.mse_loss(render_out['rgb_fine'], rgb_gt)
                depth_loss = self._compute_depth_loss(
                    render_out['depth_fine'],
                    depth_gt
                )
                
                loss = rgb_loss + self.config['model']['nerf']['depth_weight'] * depth_loss
                
                if not torch.isfinite(loss):
                    print(f"Invalid loss value: {loss.item()}")
                    return {
                        'loss': 0.0,
                        'rgb_loss': 0.0,
                        'depth_loss': 0.0
                    }
                
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.nerf_model.parameters(), 1.0)

                for name, param in self.nerf_model.named_parameters():
                    if param.grad is not None:
                        if not param.grad.isfinite().all():
                            print(f"Invalid gradients in {name}")
                            return {
                                'loss': 0.0,
                                'rgb_loss': 0.0,
                                'depth_loss': 0.0
                            }
                
                self.optimizer.step()
                
                return {
                    'loss': loss.item(),
                    'rgb_loss': rgb_loss.item(),
                    'depth_loss': depth_loss.item()
                }
                
        except RuntimeError as e:
            if "device-side assert triggered" in str(e):
                print("CUDA device assert - skipping problematic batch")
                return {
                    'loss': 0.0,
                    'rgb_loss': 0.0,
                    'depth_loss': 0.0
                }
            raise e
    
    @torch.no_grad()
    def validate(self, val_loader):
        """Validate the model."""
        self.nerf_model.eval()
        self.renderer.training = False
        total_psnr = 0
        total_depth_error = 0
        num_batches = 0
        
        max_val_batches = 5
        
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= max_val_batches:
                break

            rays_o = batch['rays_o'].to(self.device)  
            rays_d = batch['rays_d'].to(self.device)
            rgb_gt = batch['rgb'].to(self.device)
            depth_gt = batch['depth'].to(self.device)
            
            B, H, W, _ = rays_o.shape
            rays_o = rays_o.reshape(B, -1, 3)  
            rays_d = rays_d.reshape(B, -1, 3)
            rgb_gt = rgb_gt.reshape(B, -1, 3)
            depth_gt = depth_gt.reshape(B, -1)

            val_chunk_size = 4096
            pred_rgbs = []
            pred_depths = []
            
            for i in range(0, rays_o.shape[1], val_chunk_size):
                end_i = min(i + val_chunk_size, rays_o.shape[1])
                chunk_rays_o = rays_o[:, i:end_i]
                chunk_rays_d = rays_d[:, i:end_i]
                
                chunk_out = self.renderer.render_rays(
                    self.nerf_model,
                    chunk_rays_o,
                    chunk_rays_d,
                    near=0.1,
                    far=10.0
                )
                
                pred_rgbs.append(chunk_out['rgb_fine'])
                pred_depths.append(chunk_out['depth_fine'])

            pred_rgb = torch.cat(pred_rgbs, dim=1)
            pred_depth = torch.cat(pred_depths, dim=1)

            mse = F.mse_loss(pred_rgb, rgb_gt)
            psnr = -10.0 * torch.log10(mse)
            depth_error = F.l1_loss(pred_depth, depth_gt)
            
            total_psnr += psnr.item()
            total_depth_error += depth_error.item()
            num_batches += 1
        
        avg_psnr = total_psnr / num_batches if num_batches > 0 else 0
        avg_depth_error = total_depth_error / num_batches if num_batches > 0 else 0
        
        return {
            'psnr': avg_psnr,
            'depth_error': avg_depth_error
        }
        
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.nerf_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics
        }
        
        torch.save(checkpoint, os.path.join(self.log_dir, 'latest.pt'))
        
        if is_best:
            torch.save(checkpoint, os.path.join(self.log_dir, 'best.pt'))
    

    def train(self, num_epochs):
        try:
            train_loader, val_loader = get_nerf_loaders(
                self.config['data']['dir'],
                batch_size=1,
                img_wh=(256, 256),
                n_rays=2048
            )
            
            best_psnr = 0
            patience = self.config['training']['patience']
            patience_counter = 0
            
            for epoch in range(num_epochs):
                # Training
                self.nerf_model.train()
                train_loader.dataset.epoch = epoch
                
                epoch_losses = []
                pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
                
                for batch in pbar:
                    try:
                        metrics = self.train_step(batch)
                        epoch_losses.append(metrics['loss'])
                        
                        pbar.set_postfix({
                            'loss': f"{metrics['loss']:.4f}",
                            'rgb_loss': f"{metrics['rgb_loss']:.4f}",
                            'depth_loss': f"{metrics['depth_loss']:.4f}"
                        })
                        
                        self.writer.add_scalar('Loss/train', metrics['loss'], self.global_step)
                        self.writer.add_scalar('Loss/rgb', metrics['rgb_loss'], self.global_step)
                        self.writer.add_scalar('Loss/depth', metrics['depth_loss'], self.global_step)
                        
                        self.global_step += 1
                    except RuntimeError as e:
                        print(f"Skipping problematic batch: {str(e)}")
                        continue
                
                # Validation every 5 epochs
                if epoch % 5 == 0:
                    try:
                        self.nerf_model.eval()
                        with torch.no_grad():
                            val_metrics = self.validate(val_loader)
                            
                        if val_metrics is not None:
                            self.writer.add_scalar('Metrics/PSNR', val_metrics['psnr'], epoch)
                            self.writer.add_scalar('Metrics/Depth_Error', val_metrics['depth_error'], epoch)
                            
                            # Save best model
                            if val_metrics['psnr'] > best_psnr:
                                best_psnr = val_metrics['psnr']
                                patience_counter = 0
                                self.save_checkpoint(epoch, val_metrics, is_best=True)
                            else:
                                patience_counter += 1
                    except Exception as e:
                        print(f"Validation failed: {str(e)}")

                self.save_checkpoint(epoch, {'loss': np.mean(epoch_losses)}, is_best=False)
                
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch} epochs")
                    break
                
                self.scheduler.step()
                
        except Exception as e:
            print(f"Training stopped due to error: {str(e)}")
            self.save_checkpoint(epoch, {'loss': np.mean(epoch_losses)}, is_best=False)

def main():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
    with open('configs/model_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    trainer = NeRFTrainer(config)
    trainer.train(num_epochs=config['training']['epochs'])

if __name__ == '__main__':
    main()