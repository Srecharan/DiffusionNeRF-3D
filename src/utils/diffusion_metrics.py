import torch
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import numpy as np

class DiffusionDebugger:
    def __init__(self, diffusion_model):
        self.diffusion = diffusion_model
        self.metrics_history = {
            'psnr': [],
            'ssim': [],
            'edge_preservation': []
        }
        
    def compute_psnr(self, pred, target):
        mse = torch.mean((pred - target) ** 2).item()
        return -10 * torch.log10(torch.tensor(mse)).item()
        
    def compute_edge_preservation(self, pred, target):
        def get_edges(x):
            dx = x[..., 1:] - x[..., :-1]  
            dy = x[..., 1:, :] - x[..., :-1, :]  
            
            dx = dx[..., :-1, :]  # Now (B, C, H-1, W-1)
            dy = dy[..., :, :-1]  # Now (B, C, H-1, W-1)
            edges = torch.sqrt(dx**2 + dy**2)
            return edges
            
        pred_edges = get_edges(pred)
        target_edges = get_edges(target)
        
        stacked = torch.stack([
            pred_edges.reshape(-1),
            target_edges.reshape(-1)
        ])

        correlation = torch.corrcoef(stacked)[0, 1].item()
        
        return correlation

    def evaluate_step(self, original, noisy, denoised):
        original_np = original.cpu().numpy().squeeze(1)  
        denoised_np = denoised.cpu().numpy().squeeze(1)  
        
        ssim_vals = []
        for i in range(len(original_np)):
            ssim_val = ssim(
                original_np[i],  
                denoised_np[i], 
                data_range=2.0,  
                win_size=3,      
                channel_axis=None 
            )
            ssim_vals.append(ssim_val)
        
        ssim_val = np.mean(ssim_vals)
        psnr = self.compute_psnr(denoised, original)
        edge_score = self.compute_edge_preservation(denoised, original)

        self.metrics_history['psnr'].append(psnr)
        self.metrics_history['ssim'].append(ssim_val)
        self.metrics_history['edge_preservation'].append(edge_score)
        
        return {
            'psnr': psnr,
            'ssim': ssim_val,
            'edge_preservation': edge_score
        }
        
    def plot_metrics(self, save_path='debug_outputs/metrics.png'):
        plt.figure(figsize=(15, 5))
        
        # Plot PSNR
        plt.subplot(131)
        plt.plot(self.metrics_history['psnr'])
        plt.title('PSNR over Steps')
        plt.xlabel('Step')
        plt.ylabel('PSNR (dB)')
        
        # Plot SSIM
        plt.subplot(132)
        plt.plot(self.metrics_history['ssim'])
        plt.title('SSIM over Steps')
        plt.xlabel('Step')
        plt.ylabel('SSIM')
        
        # Plot Edge Preservation
        plt.subplot(133)
        plt.plot(self.metrics_history['edge_preservation'])
        plt.title('Edge Preservation')
        plt.xlabel('Step')
        plt.ylabel('Correlation')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

# Usage example:
"""
debugger = DiffusionDebugger(diffusion_model)

# During training/testing
metrics = debugger.evaluate_step(original_depth, noisy_depth, denoised_depth)
print(f"PSNR: {metrics['psnr']:.2f}dB, SSIM: {metrics['ssim']:.3f}")

# Plot metrics
debugger.plot_metrics()
"""