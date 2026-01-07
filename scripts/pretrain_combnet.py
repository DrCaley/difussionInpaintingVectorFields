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

from data_prep.data_initializer import DDInitializer
from ddpm.vector_combination.combiner_unet import VectorCombinationUNet
from ddpm.vector_combination.combination_loss import PhysicsInformedLoss


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
    save_path=None
):
    """
    Pretrain the CombNet on pre-generated realistic samples.
    """
    print("=" * 60)
    print("PRETRAINING COMBNET")
    print("=" * 60)
    
    # Setup
    dd = DDInitializer()
    device = dd.get_device()
    
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
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Loss function
    loss_fn = PhysicsInformedLoss(
        weight_fidelity=dd.get_attribute("fidelity_weight"),
        weight_physics=dd.get_attribute("physics_weight"),
        weight_smooth=dd.get_attribute("smooth_weight")
    ).to(device)
    
    print(f"Training on device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Samples: {len(tensor_dataset)}")
    print(f"Loss weights - fidelity: {loss_fn.weights['fidelity']}, physics: {loss_fn.weights['physics']}, smooth: {loss_fn.weights['smooth']}")
    print()
    
    best_loss = float('inf')
    train_losses = []
    
    for epoch in range(epochs):
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
            batch_loss = 0
            for i in range(batch_inputs.shape[0]):
                prediction = model(batch_inputs[i:i+1])
                loss, stats = loss_fn(
                    prediction, 
                    batch_known[i:i+1], 
                    batch_inpainted[i:i+1], 
                    batch_mask[i:i+1]
                )
                batch_loss = batch_loss + loss
            
            batch_loss = batch_loss / batch_inputs.shape[0]
            
            # Backward pass
            batch_loss.backward()
            optimizer.step()
            
            epoch_losses.append(batch_loss.item())
            pbar.set_postfix(loss=f"{batch_loss.item():.4f}")
        
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
    args = parser.parse_args()
    
    pretrain_combnet(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
