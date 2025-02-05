# src/data/nyu_dataset.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class NYUDepthDataset(Dataset):
    def __init__(self, root_dir, transform=None, split='train'):
        """
        Args:
            root_dir (str): Directory with nyu_depth_v2 data
            transform (callable, optional): Optional transform to be applied on samples
            split (str): 'train' or 'val' split
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Get all file names
        self.rgb_dir = os.path.join(root_dir, 'rgb')
        self.depth_dir = os.path.join(root_dir, 'depth')
        
        # Get all files and sort them
        self.files = sorted(os.listdir(self.rgb_dir))
        
        # Split dataset (80% train, 20% val)
        split_idx = int(len(self.files) * 0.8)
        if split == 'train':
            self.files = self.files[:split_idx]
        else:
            self.files = self.files[split_idx:]

    def __len__(self):
        return len(self.files)


    def __getitem__(self, idx):
        # Get current frame
        current = self._load_frame(idx)
        
        # Get adjacent frame if available
        next_idx = min(idx + 1, len(self.files) - 1)
        next_frame = self._load_frame(next_idx)
        
        return {
            'depth': current['depth'],
            'rgb': current['rgb'],
            'next_depth': next_frame['depth'],
            'next_rgb': next_frame['rgb'],
            'pose': self._compute_relative_pose(idx, next_idx)  # You'll need to implement this
        }

def get_data_loaders(root_dir, batch_size=32, num_workers=4):
    """Create train and validation data loaders"""
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to uniform size
    ])
    
    # Create datasets
    train_dataset = NYUDepthDataset(root_dir, transform=transform, split='train')
    val_dataset = NYUDepthDataset(root_dir, transform=transform, split='val')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

# In nyu_dataset.py, add these methods
def _load_frame(self, idx):
    """Load a single frame of data"""
    rgb_path = os.path.join(self.rgb_dir, self.files[idx])
    depth_path = os.path.join(self.depth_dir, self.files[idx])
    
    rgb = np.load(rgb_path)
    depth = np.load(depth_path)
    
    # Convert to torch tensors
    rgb = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
    depth = torch.from_numpy(depth).float().unsqueeze(0)
    
    if self.transform:
        rgb = self.transform(rgb)
        depth = self.transform(depth)
    
    return {'rgb': rgb, 'depth': depth}

def _compute_relative_pose(self, idx1, idx2):
    """Compute relative pose between two frames"""
    # For NYU dataset, we'll use identity transform as baseline
    # In a real implementation, this would use actual camera poses
    return torch.eye(4, dtype=torch.float32)