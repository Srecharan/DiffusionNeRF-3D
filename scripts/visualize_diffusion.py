# scripts/visualize_diffusion.py

import torch
import torch.nn.functional as F
from torch.amp import autocast
import matplotlib.pyplot as plt
import sys
import os
from tqdm import tqdm
import yaml

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.models.unet import UNet
from src.models.diffusion import DiffusionModel
from src.data.nyu_dataset import get_data_loaders

# scripts/visualize_diffusion.py

def load_model(checkpoint_path, config, device):
    """Load trained model from checkpoint"""
    model = UNet(
        in_channels=config['model']['unet']['in_channels'],
        out_channels=config['model']['unet']['out_channels'],
        time_emb_dim=config['model']['unet']['time_emb_dim'],
        base_channels=config['model']['unet']['base_channels'],
        attention=config['model']['unet']['attention']
    ).to(device)
    
    diffusion = DiffusionModel(
        model,
        n_steps=config['diffusion']['n_steps'],
        beta_start=float(config['diffusion']['beta_start']),
        beta_end=float(config['diffusion']['beta_end']),
        beta_schedule=config['diffusion']['beta_schedule'],
        device=device
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Process state dict to remove _orig_mod prefix
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    
    for key, value in state_dict.items():
        if key.startswith('_orig_mod.'):
            new_key = key.replace('_orig_mod.', '')
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    # Load the processed state dict
    model.load_state_dict(new_state_dict)
    model.eval()  # Set to evaluation mode
    
    return model, diffusion

def visualize_samples(model, diffusion, val_loader, device, num_samples=5):
    """Visualize original, noisy, and denoised depth maps"""
    os.makedirs('visualizations', exist_ok=True)
    model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= num_samples:
                break
                
            depth_maps = batch['depth'].to(device)
            
            # Add noise
            t = torch.ones(depth_maps.shape[0], device=device).long() * (diffusion.n_steps // 2)
            noisy_depths, noise = diffusion.noise_images(depth_maps, t)
            
            # Denoise
            with autocast(device_type='cuda', enabled=True):
                noise_pred = model(noisy_depths, t/diffusion.n_steps)
            
            # Denoise the depth maps
            denoised = noisy_depths - noise_pred
            
            # Calculate error map
            error = torch.abs(denoised - depth_maps)
            
            # Visualize with error map
            fig, axes = plt.subplots(2, 2, figsize=(15, 15))
            
            # Original depth map
            im0 = axes[0,0].imshow(depth_maps[0, 0].cpu(), cmap='plasma')
            axes[0,0].set_title('Original Depth')
            plt.colorbar(im0, ax=axes[0,0])
            
            # Noisy depth map
            im1 = axes[0,1].imshow(noisy_depths[0, 0].cpu(), cmap='plasma')
            axes[0,1].set_title('Noisy Depth')
            plt.colorbar(im1, ax=axes[0,1])
            
            # Denoised prediction
            im2 = axes[1,0].imshow(denoised[0, 0].cpu(), cmap='plasma')
            axes[1,0].set_title('Denoised Depth')
            plt.colorbar(im2, ax=axes[1,0])
            
            # Error map
            im3 = axes[1,1].imshow(error[0, 0].cpu(), cmap='hot')
            axes[1,1].set_title('Error Map')
            plt.colorbar(im3, ax=axes[1,1])
            
            plt.tight_layout()
            plt.savefig(f'visualizations/sample_{i}_detailed.png', dpi=300, bbox_inches='tight')
            plt.close()

def calculate_metrics(model, diffusion, val_loader, device):
    """Calculate quantitative metrics"""
    model.eval()
    total_mse = 0
    total_psnr = 0
    num_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Calculating metrics"):
            depth_maps = batch['depth'].to(device)
            
            # Add noise
            t = torch.ones(depth_maps.shape[0], device=device).long() * (diffusion.n_steps // 2)
            noisy_depths, noise = diffusion.noise_images(depth_maps, t)
            
            # Denoise
            with autocast(device_type='cuda', enabled=True):  # Fixed autocast
                noise_pred = model(noisy_depths, t/diffusion.n_steps)
            
            # Calculate metrics
            denoised = noisy_depths - noise_pred
            mse = F.mse_loss(denoised, depth_maps)
            psnr = -10 * torch.log10(mse)
            
            total_mse += mse.item() * depth_maps.shape[0]
            total_psnr += psnr.item() * depth_maps.shape[0]
            num_samples += depth_maps.shape[0]
    
    avg_mse = total_mse / num_samples
    avg_psnr = total_psnr / num_samples
    
    return {'MSE': avg_mse, 'PSNR': avg_psnr}

def get_latest_experiment_dir():
    """Get the most recent experiment directory"""
    runs_dir = 'runs'
    experiment_dirs = [os.path.join(runs_dir, d) for d in os.listdir(runs_dir) 
                      if d.startswith('diffusion_experiment_')]
    if not experiment_dirs:
        raise RuntimeError("No experiment directories found!")
    
    # Sort by creation time and get the most recent
    latest_dir = max(experiment_dirs, key=os.path.getctime)
    return latest_dir

def main():
    print("Starting visualization and evaluation...")
    
    # Load config
    with open('configs/model_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get the latest experiment directory automatically
    experiment_dir = get_latest_experiment_dir()
    print(f"Using latest experiment directory: {experiment_dir}")
    
    # Load best model
    checkpoint_path = os.path.join(experiment_dir, 'best_model.pt')
    print(f"Loading model from: {checkpoint_path}")
    
    model, diffusion = load_model(checkpoint_path, config, device)
    
    # Create data loader
    data_dir = 'data/raw/nyu_depth_v2'
    _, val_loader = get_data_loaders(data_dir, batch_size=4)
    
    # Create visualization directory
    os.makedirs('visualizations', exist_ok=True)
    
    # Generate visualizations
    print("Generating visualizations...")
    visualize_samples(model, diffusion, val_loader, device)
    
    # Calculate metrics
    print("Calculating metrics...")
    metrics = calculate_metrics(model, diffusion, val_loader, device)
    
    # Print and save metrics
    print("\nMetrics on validation set:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    with open('visualizations/metrics.txt', 'w') as f:
        for metric_name, value in metrics.items():
            f.write(f"{metric_name}: {value:.4f}\n")
    
    print("\nVisualization completed! Check the 'visualizations' directory for results.")

if __name__ == '__main__':
    main()