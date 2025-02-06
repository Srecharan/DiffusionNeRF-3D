# scripts/train_diffusion.py

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.amp as amp
from tqdm import tqdm
import sys
import os
import yaml
from datetime import datetime

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.models.unet import UNet
from src.models.diffusion import DiffusionModel, MultiViewDiffusion
from src.data.nyu_dataset import get_data_loaders
from src.models.feature_extractor import FeatureExtractor 

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def create_scheduler(optimizer, config, num_training_steps):
    from torch.optim.lr_scheduler import OneCycleLR, LinearLR, SequentialLR
    
    warmup_epochs = config['training']['scheduler']['warmup_epochs']
    total_epochs = config['training']['epochs']

    warmup_steps = max(1, int(warmup_epochs * num_training_steps / total_epochs))
    remaining_steps = max(1, num_training_steps - warmup_steps)
    
    print(f"Scheduler setup: {num_training_steps} total steps, {warmup_steps} warmup steps, {remaining_steps} remaining steps")

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps
    )

    main_scheduler = OneCycleLR(
        optimizer,
        max_lr=config['training']['learning_rate'],
        total_steps=remaining_steps,
        pct_start=0.3,
        anneal_strategy='cos',
        final_div_factor=config['training']['scheduler']['min_lr'] / config['training']['learning_rate'],
        div_factor=10.0
    )
    
    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_steps]
    )

def train(config):
    torch.cuda.set_per_process_memory_fraction(0.9)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join('runs', f'diffusion_experiment_{timestamp}')
    writer = SummaryWriter(log_dir)
    
    train_loader, val_loader = get_data_loaders(
        config['data']['dir'],  # Updated to match new config structure
        batch_size=config['training']['batch_size']
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(
        in_channels=config['model']['unet']['in_channels'],
        out_channels=config['model']['unet']['out_channels'],
        time_emb_dim=config['model']['unet']['time_emb_dim'],
        base_channels=config['model']['unet']['base_channels'],
        attention=config['model']['unet']['attention']
    ).to(device)

    model = torch.compile(model)
    
    diffusion = MultiViewDiffusion(
        model,
        n_steps=config['diffusion']['n_steps'],
        beta_start=config['diffusion']['beta_start'],
        beta_end=config['diffusion']['beta_end'],
        beta_schedule=config['diffusion']['beta_schedule'],
        device=device
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * config['training']['epochs']
    print(f"Training setup: {steps_per_epoch} steps per epoch, {total_steps} total steps")

    scheduler = create_scheduler(optimizer, config, total_steps)

    scaler = amp.GradScaler('cuda')
    
    early_stopping = EarlyStopping(patience=config['training']['patience'])

    best_val_loss = float('inf')

    for epoch in range(config['training']['epochs']):
        model.train()
        total_train_loss = 0
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["training"]["epochs"]}') as pbar:
            for batch in pbar:
                depth_maps = batch['depth'].to(device)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                with amp.autocast('cuda'):
                    optimizer.zero_grad(set_to_none=True)
                    loss = diffusion.training_step(depth_maps)
                
                scaler.scale(loss).backward()

                if config['training']['clip_value'] > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        config['training']['clip_value']
                    )
                
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                
                total_train_loss += loss.item()

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.6f}'
                })

                if torch.cuda.is_available():
                    writer.add_scalar(
                        'Memory/allocated',
                        torch.cuda.memory_allocated() / 1024**2,
                        epoch * len(train_loader) + pbar.n
                    )
        
        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                depth_maps = batch['depth'].to(device)
                with amp.autocast('cuda'):
                    loss = diffusion.validation_step(depth_maps)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)

        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Learning_rate', scheduler.get_last_lr()[0], epoch)
        
        print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': config,
            }, os.path.join(log_dir, 'best_model.pt'))

        if (epoch + 1) % config['training']['save_every'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': config,
            }, os.path.join(log_dir, f'checkpoint_epoch_{epoch+1}.pt'))

        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

if __name__ == '__main__':
    try:
        with open('configs/model_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
            
        data_dir = config['data']['dir']
        if not os.path.exists(data_dir):
            alt_data_dir = os.path.join(project_root, data_dir)
            if os.path.exists(alt_data_dir):
                config['data']['dir'] = alt_data_dir
            else:
                raise FileNotFoundError(f"Data directory not found at {data_dir} or {alt_data_dir}")
        
        os.makedirs('runs', exist_ok=True)
        
        print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        train(config)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise