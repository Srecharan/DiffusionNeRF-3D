# scripts/download_nyu.py

import os
import gdown
import numpy as np
import h5py
from tqdm import tqdm

def download_nyu_dataset(target_dir):
    """Download NYU Depth V2 labeled dataset"""
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # URL for the labeled dataset
    url = 'http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat'
    output = os.path.join(target_dir, 'nyu_depth_v2_labeled.mat')
    
    if not os.path.exists(output):
        print("Downloading NYU Depth V2 dataset...")
        gdown.download(url, output, quiet=False)
    
    return output

def process_nyu_dataset(mat_file, output_dir):
    """Process the .mat file into numpy arrays"""
    print("Processing NYU Depth V2 dataset...")
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'rgb'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'depth'), exist_ok=True)
    
    # Load the .mat file
    dataset = h5py.File(mat_file, 'r')
    
    # Get the data
    images = dataset['images']
    depths = dataset['depths']
    
    for i in tqdm(range(len(images))):
        # Save RGB image
        img = np.transpose(images[i], (2, 1, 0))
        np.save(os.path.join(output_dir, 'rgb', f'{i:05d}.npy'), img)
        
        # Save depth map
        depth = np.transpose(depths[i])
        np.save(os.path.join(output_dir, 'depth', f'{i:05d}.npy'), depth)

def main():
    target_dir = 'data/raw/nyu_depth_v2'
    mat_file = download_nyu_dataset(target_dir)
    process_nyu_dataset(mat_file, target_dir)

if __name__ == '__main__':
    main()