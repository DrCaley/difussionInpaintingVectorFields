#!/usr/bin/env python
"""
Generate training dataset for CombNet by merging pairs of divergence-free fields.

The CombNet learns to fix the boundary discontinuity when two div-free fields 
are naively stitched together. We don't need actual DDPM outputs - just pairs
of div-free noise fields merged with various masks.

Usage:
    python scripts/generate_combnet_data.py
    
Output:
    results/combnet_training_data.pt
"""
import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from data_prep.data_initializer import DDInitializer
from ddpm.utils.noise_utils import DivergenceFreeNoise
from ddpm.Testing.model_inpainter import StraightLineMaskGenerator


def generate_dataset(
    num_field_pairs=5000,     # Number of div-free field pairs to generate
    masks_per_pair=8,         # Number of different masks per field pair
    save_path=None
):
    """
    Generate CombNet training data by merging pairs of divergence-free fields.
    
    For each field pair:
    1. Generate two independent div-free noise fields
    2. Apply MULTIPLE random masks to get many samples from one pair
    3. Naive merge creates the boundary discontinuity
    4. Save (known, inpainted, mask, naive) tuples
    """
    print("=" * 60)
    print("GENERATING COMBNET TRAINING DATA")
    print("=" * 60)
    
    # Setup
    dd = DDInitializer()
    device = dd.get_device()
    
    if save_path is None:
        save_path = BASE_DIR / "results" / "combnet_training_data.pt"
    
    # Noise strategy (div-free)
    noise_strategy = DivergenceFreeNoise()
    
    # Field shape
    channels, height, width = 2, 128, 64
    field_shape = (1, channels, height, width)
    
    # Mask generators - variety of widths and positions
    mask_generators = [
        StraightLineMaskGenerator(1, 1),
        StraightLineMaskGenerator(2, 1),
        StraightLineMaskGenerator(3, 1),
        StraightLineMaskGenerator(4, 1),
        StraightLineMaskGenerator(5, 1),
    ]
    
    # Also load real ocean data for land masks
    train_data = dd.get_training_data()
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    data_iter = iter(train_loader)
    
    # Storage
    dataset = {
        'known': [],
        'inpainted': [],
        'mask': [],
        'naive': [],
    }
    
    total_samples = num_field_pairs * masks_per_pair
    print(f"Generating data:")
    print(f"  Field pairs: {num_field_pairs}")
    print(f"  Masks per pair: {masks_per_pair}")
    print(f"  Total samples: ~{total_samples}")
    print(f"  Field shape: {field_shape}")
    print(f"  Device: {device}")
    print()
    
    for pair_idx in tqdm(range(num_field_pairs), desc="Generating field pairs"):
        # Get a real ocean field for land mask
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        
        real_field = batch[0].to(device)
        real_field_unstd = dd.get_standardizer().unstandardize(real_field.squeeze(0)).unsqueeze(0)
        
        # Get actual field shape from real data
        actual_shape = real_field_unstd.shape
        land_mask = (real_field_unstd.abs() > 1e-5).float()
        
        # Generate two independent div-free noise fields (once per pair)
        field_a = noise_strategy(torch.zeros(actual_shape, device=device), None)
        field_b = noise_strategy(torch.zeros(actual_shape, device=device), None)
        
        # Apply land mask (zero out land regions)
        field_a = field_a * land_mask
        field_b = field_b * land_mask
        
        # Apply multiple different masks to this field pair
        for mask_idx in range(masks_per_pair):
            # Random mask generator (different position each time)
            mask_gen = np.random.choice(mask_generators)
            raw_mask = mask_gen.generate_mask(actual_shape).to(device)
            mask = raw_mask * land_mask
            
            # Skip if mask doesn't cover enough area
            if mask.sum() < 10:
                continue
            
            # Naive merge (this has a boundary discontinuity)
            naive = field_a * (1 - mask) + field_b * mask
            
            # Store
            dataset['known'].append(field_a.cpu())
            dataset['inpainted'].append(field_b.cpu())
            dataset['mask'].append(mask.cpu())
            dataset['naive'].append(naive.cpu())
    
    # Stack into tensors
    print(f"\nCollected {len(dataset['known'])} samples")
    
    final_dataset = {
        'known': torch.cat(dataset['known'], dim=0),
        'inpainted': torch.cat(dataset['inpainted'], dim=0),
        'mask': torch.cat(dataset['mask'], dim=0),
        'naive': torch.cat(dataset['naive'], dim=0),
    }
    
    print(f"Dataset shapes:")
    for key, val in final_dataset.items():
        print(f"  {key}: {val.shape}")
    
    # Save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(final_dataset, save_path)
    print(f"\n✅ Saved dataset to: {save_path}")
    
    return final_dataset


if __name__ == '__main__':
    generate_dataset(
        num_field_pairs=5000,   # 5000 unique div-free field pairs
        masks_per_pair=8,       # 8 different masks per pair
        # Total: ~40,000 samples
    )
