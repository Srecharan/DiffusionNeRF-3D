import sys
import os
import torch
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.data.nyu_dataset import get_data_loaders
from src.models.depth_refiner import SimpleDepthRefiner

def test_depth_refiner():
    print("Starting depth refiner test...")
    
    with open('configs/model_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    config['training']['batch_size'] = 2
    
    print("\n1. Testing Data Loading...")
    train_loader, val_loader = get_data_loaders(
        root_dir=config['data']['dir'],
        config=config,
        num_workers=0
    )

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
    
    print("\n2. Setting up model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = SimpleDepthRefiner(in_channels=1, base_channels=32).to(device)
    
    print("\n3. Testing Forward Pass...")
    with torch.no_grad():
        depth = depth.to(device)
        noisy_depth = depth + 0.1 * torch.randn_like(depth)
        refined_depth = model(noisy_depth)
        
        loss = model.loss_fn(refined_depth, depth)
        print(f"Test loss: {loss.item():.4f}")

        plt.figure(figsize=(15, 5))
        
        plt.subplot(131)
        plt.imshow(depth[0, 0].cpu().numpy(), cmap='plasma')
        plt.colorbar()
        plt.title('Original Depth')
        
        plt.subplot(132)
        plt.imshow(noisy_depth[0, 0].cpu().numpy(), cmap='plasma')
        plt.colorbar()
        plt.title('Noisy Depth')
        
        plt.subplot(133)
        plt.imshow(refined_depth[0, 0].cpu().numpy(), cmap='plasma')
        plt.colorbar()
        plt.title('Refined Depth')
        
        os.makedirs('debug_outputs', exist_ok=True)
        plt.savefig('debug_outputs/depth_refiner_test.png')
        plt.close()

        edges = model.compute_edges(depth)
        refined_edges = model.compute_edges(refined_depth)
        
        plt.figure(figsize=(10, 5))
        
        plt.subplot(121)
        plt.imshow(edges[0, 0].cpu().numpy(), cmap='gray')
        plt.colorbar()
        plt.title('Original Edges')
        
        plt.subplot(122)
        plt.imshow(refined_edges[0, 0].cpu().numpy(), cmap='gray')
        plt.colorbar()
        plt.title('Refined Edges')
        
        plt.savefig('debug_outputs/depth_refiner_edges.png')
        plt.close()
    
    print("\n4. Memory Usage...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    if torch.cuda.is_available():
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
        print(f"GPU Memory cached: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")
    
    print("\nTest completed! Check 'debug_outputs' directory for visualizations.")

if __name__ == '__main__':
    test_depth_refiner()