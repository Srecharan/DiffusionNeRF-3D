import sys
import os
import torch
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.data.nyu_dataset import get_data_loaders
from src.models.unet import UNet
from src.models.diffusion import DiffusionModel  # Note the changed import
from src.utils.diffusion_metrics import DiffusionDebugger  # New import

def analyze_data():
    print("Starting enhanced debug analysis...")
    
    # Load config
    with open('configs/model_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Override batch size for testing
    config['training']['batch_size'] = 2
    
    print("\n1. Testing Data Loading...")
    train_loader, val_loader = get_data_loaders(
        root_dir=config['data']['dir'],
        config=config,
        num_workers=0
    )
    
    # Get a sample batch
    sample_batch = next(iter(train_loader))
    depth = sample_batch['depth']
    
    print("\nData Analysis:")
    print("-" * 50)
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Val dataset size: {len(val_loader.dataset)}")
    print(f"Batch shape: {depth.shape}")
    print(f"Depth range: [{depth.min():.3f}, {depth.max():.3f}]")
    print(f"Depth mean: {depth.mean():.3f}")
    print(f"Depth std: {depth.std():.3f}")
    
    # Basic visualization
    visualize_depth_data(depth, 'debug_outputs/depth_analysis.png')
    
    print("\n2. Testing Model Setup...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = UNet(
        in_channels=1,
        out_channels=1,
        time_emb_dim=256  # using default value or change as needed
    ).to(device)
    
    # Create improved diffusion
    diffusion = DiffusionModel(
        model,
        n_steps=config['diffusion']['n_steps'],
        beta_schedule='cosine',  # changed from 'improved_cosine'
        beta_start=1e-6,
        beta_end=0.005,
        device=device
    )
    
    # Create debugger
    debugger = DiffusionDebugger(diffusion)
    
    print("\n3. Testing Forward Pass with Enhanced Monitoring...")
    with torch.no_grad():
        depth = depth.to(device)
        
        # Test multiple noise levels
        noise_levels = [0, diffusion.n_steps // 4, diffusion.n_steps // 2, 3 * diffusion.n_steps // 4]
        
        plt.figure(figsize=(15, 5 * len(noise_levels)))
        for idx, t_step in enumerate(noise_levels):
            # Ensure all tensors are on the same device
            depth = depth.to(device)
            t = torch.ones(depth.shape[0], device=device, dtype=torch.long) * t_step
            
            # Test noising process
            noisy_depth, noise = diffusion.noise_images(depth, t)
            
            # Test denoising
            pred = model(noisy_depth, t)
            
            # Compute metrics
            metrics = debugger.evaluate_step(depth, noisy_depth, pred)
            print(f"\nMetrics for noise level {t_step}:")
            print(f"PSNR: {metrics['psnr']:.2f}dB")
            print(f"SSIM: {metrics['ssim']:.3f}")
            print(f"Edge Preservation: {metrics['edge_preservation']:.3f}")
            
            # Visualize
            plt.subplot(len(noise_levels), 3, 3*idx + 1)
            plt.imshow(depth[0, 0].cpu().numpy(), cmap='plasma')
            plt.colorbar()
            plt.title(f'Original Depth (t={t_step})')
            
            plt.subplot(len(noise_levels), 3, 3*idx + 2)
            plt.imshow(noisy_depth[0, 0].cpu().numpy(), cmap='plasma')
            plt.colorbar()
            plt.title(f'Noisy Depth (t={t_step})')
            
            plt.subplot(len(noise_levels), 3, 3*idx + 3)
            plt.imshow(pred[0, 0].cpu().numpy(), cmap='plasma')
            plt.colorbar()
            plt.title(f'Denoised Depth (t={t_step})')
        
        plt.tight_layout()
        plt.savefig('debug_outputs/diffusion_process_enhanced.png')
        plt.close()
        
        # Plot metrics
        debugger.plot_metrics()
    
    print("\n4. Testing Memory Usage...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    if torch.cuda.is_available():
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
        print(f"GPU Memory cached: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")
    
    print("\nEnhanced debug analysis completed! Check 'debug_outputs' directory for visualizations.")

def visualize_depth_data(depth, save_path):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(depth[0, 0].numpy(), cmap='plasma')
    plt.colorbar()
    plt.title('Sample Depth Map 1')
    
    plt.subplot(132)
    plt.imshow(depth[1, 0].numpy(), cmap='plasma')
    plt.colorbar()
    plt.title('Sample Depth Map 2')
    
    plt.subplot(133)
    plt.hist(depth.numpy().flatten(), bins=50)
    plt.title('Depth Value Distribution')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':
    analyze_data()