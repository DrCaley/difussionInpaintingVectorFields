#!/usr/bin/env python
"""
Pretrain the CombNet (boundary divergence fixer) on pre-generated dataset.

First run: python scripts/generate_combnet_data.py
Then run:  python scripts/pretrain_combnet.py

The trained model will be saved to: ddpm/Trained_Models/pretrained_combnet.pt
"""
import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
import yaml

from ddpm.vector_combination.combiner_unet import VectorCombinationUNet
from ddpm.vector_combination.combination_loss import PhysicsInformedLoss


def get_device():
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_config():
    """Load configuration from data.yaml."""
    config_path = BASE_DIR / "data.yaml"
    if not config_path.exists():
        # Return defaults if no config file
        return {
            'fidelity_weight': 0.01,
            'physics_weight': 2.0,
            'smooth_weight': 0.0,
        }

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def load_dataset(data_path=None):
    """Load pre-generated CombNet training data."""
    if data_path is None:
        data_path = BASE_DIR / "results" / "combnet_training_data.pt"
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}\n"
            "Run 'python scripts/generate_combnet_data.py' first to generate training data."
        )
    
    dataset = torch.load(data_path, weights_only=False)
    print(f"Loaded dataset from {data_path}")
    print(f"  Samples: {len(dataset['known'])}")
    
    return dataset


def pretrain_combnet(
    epochs=20,
    batch_size=8,
    lr=1e-3,
    data_path=None,
    save_path=None,
    resume=False,
    fresh_start=False
):
    """
    Pretrain the CombNet on pre-generated realistic samples.
    """
    print("=" * 60)
    print("PRETRAINING COMBNET")
    print("=" * 60)

    # Setup device and config
    device = get_device()
    config = load_config()
    
    if save_path is None:
        save_path = BASE_DIR / "ddpm" / "Trained_Models" / "pretrained_combnet.pt"
    
    # Load dataset
    dataset = load_dataset(data_path)
    
    # Create DataLoader
    # Input: [naive (2ch), mask (2ch)] = 4 channels
    naive = dataset['naive']  # [N, 2, H, W]
    mask = dataset['mask']    # [N, 2, H, W]
    known = dataset['known']  # [N, 2, H, W]
    inpainted = dataset['inpainted']  # [N, 2, H, W]
    
    # Combine into input tensor
    inputs = torch.cat([naive, mask], dim=1)  # [N, 4, H, W]
    
    tensor_dataset = TensorDataset(inputs, known, inpainted, mask)
    train_loader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    model = VectorCombinationUNet(n_channels=4, n_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Use eta_min to prevent LR from going too low
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)
    
    # Resume from checkpoint if requested
    start_epoch = 0
    if (resume or fresh_start) and save_path.exists():
        print(f"Loading checkpoint from {save_path}...")
        checkpoint = torch.load(save_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if fresh_start:
            # Fresh start: keep weights but reset optimizer and scheduler
            print("  Fresh start: keeping model weights, resetting optimizer and scheduler")
            print(f"  Starting fresh with LR={lr}, will decay to eta_min=1e-4")
        else:
            # Full resume: restore optimizer state and advance scheduler
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint['loss']
            train_losses = checkpoint.get('train_losses', [])
            # Advance scheduler to correct position
            for _ in range(start_epoch):
                scheduler.step()
            print(f"  Resuming from epoch {start_epoch + 1}, best loss = {best_loss:.6f}")
    
    # Loss function
    divergence_threshold = config.get('divergence_threshold', 0.06)
    loss_fn = PhysicsInformedLoss(
        weight_fidelity=config.get('fidelity_weight', 1.0),
        weight_physics=config.get('physics_weight', 1.0),
        weight_smooth=config.get('smooth_weight', 0.0),
        divergence_threshold=divergence_threshold
    ).to(device)
    
    print(f"Training on device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Samples: {len(tensor_dataset)}")
    print(f"Loss weights - fidelity: {loss_fn.weights['fidelity']}, physics: {loss_fn.weights['physics']}, smooth: {loss_fn.weights['smooth']}")
    print(f"Divergence threshold: {divergence_threshold}")
    print()
    
    if not resume or not save_path.exists():
        best_loss = float('inf')
        train_losses = []
    
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_inputs, batch_known, batch_inpainted, batch_mask in pbar:
            batch_inputs = batch_inputs.to(device)
            batch_known = batch_known.to(device)
            batch_inpainted = batch_inpainted.to(device)
            batch_mask = batch_mask.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass (process each sample individually for loss calculation)
            batch_loss = 0.0  # Use float, not tensor
            for i in range(batch_inputs.shape[0]):
                prediction = model(batch_inputs[i:i+1])
                loss, stats = loss_fn(
                    prediction, 
                    batch_known[i:i+1], 
                    batch_inpainted[i:i+1], 
                    batch_mask[i:i+1]
                )
                # Backward immediately for each sample to free graph memory
                loss.backward()
                batch_loss += loss.item()  # Store scalar, not tensor
            
            batch_loss = batch_loss / batch_inputs.shape[0]
            
            # Step optimizer after all samples processed
            optimizer.step()
            
            epoch_losses.append(batch_loss)
            pbar.set_postfix(loss=f"{batch_loss:.4f}")
        
        # Epoch summary
        avg_loss = np.mean(epoch_losses)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.6f}, LR = {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': best_loss,
                'train_losses': train_losses,
                'config': {
                    'n_channels': 4,
                    'n_classes': 2,
                    'fidelity_weight': loss_fn.weights['fidelity'],
                    'physics_weight': loss_fn.weights['physics'],
                    'smooth_weight': loss_fn.weights['smooth'],
                }
            }, save_path)
            print(f"  ✅ Saved best model (loss={best_loss:.6f})")
        
        scheduler.step()
    
    print()
    print("=" * 60)
    print(f"Training complete! Best loss: {best_loss:.6f}")
    print(f"Model saved to: {save_path}")
    print("=" * 60)
    
    return model, train_losses


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--data', type=str, default=None, help='Path to training data (default: results/combnet_training_data.pt)')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--fresh_start', action='store_true', help='Load weights from checkpoint but reset optimizer/scheduler')
    args = parser.parse_args()
    
    data_path = Path(args.data) if args.data else None
    
    pretrain_combnet(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        data_path=data_path,
        resume=args.resume,
        fresh_start=args.fresh_start
    )
