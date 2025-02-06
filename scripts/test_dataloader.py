# scripts/test_dataloader.py

import sys
import os
import torch
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.data.nyu_dataset import get_data_loaders

def visualize_batch(batch):
    """Visualize a batch of data"""
    rgb = batch['rgb']
    depth = batch['depth']
    
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    
    for i in range(4):
        axes[0, i].imshow(rgb[i].permute(1, 2, 0))
        axes[0, i].axis('off')
        axes[0, i].set_title(f'RGB {i+1}')
        
        axes[1, i].imshow(depth[i][0], cmap='plasma')
        axes[1, i].axis('off')
        axes[1, i].set_title(f'Depth {i+1}')
    
    plt.tight_layout()
    plt.show()

def main():
    data_dir = 'data/raw/nyu_depth_v2'
    train_loader, val_loader = get_data_loaders(data_dir, batch_size=4)

    batch = next(iter(train_loader))
    visualize_batch(batch)
    
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Validation dataset size: {len(val_loader.dataset)}")
    
if __name__ == '__main__':
    main()