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
from tqdm import tqdm
import numpy as np
import random


def get_device():
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def generate_divergence_free_noise(shape, device):
    """
    Generate divergence-free noise using stream function approach.

    For a 2D vector field to be divergence-free: ∂u/∂x + ∂v/∂y = 0
    We can achieve this by using a stream function ψ:
        u = -∂ψ/∂y
        v = ∂ψ/∂x
    """
    _, _, h, w = shape

    # Generate random stream function
    psi = torch.randn(1, 1, h, w, device=device)

    # Compute derivatives to get divergence-free field
    # u = -∂ψ/∂y
    u = torch.zeros(1, 1, h, w, device=device)
    u[:, :, :-1, :] = -(psi[:, :, 1:, :] - psi[:, :, :-1, :])

    # v = ∂ψ/∂x
    v = torch.zeros(1, 1, h, w, device=device)
    v[:, :, :, :-1] = psi[:, :, :, 1:] - psi[:, :, :, :-1]

    # Combine into 2-channel vector field
    field = torch.cat([u, v], dim=1)

    return field


class SimpleStraightLineMaskGenerator:
    """Standalone straight line mask generator without DDInitializer dependency."""
    def __init__(self, num_lines=10, line_thickness=5):
        self.num_lines = num_lines
        self.line_thickness = line_thickness

    def generate_mask(self, image_shape):
        _, _, h, w = image_shape
        mask = np.ones((h, w), dtype=np.float32)

        # Generate horizontal lines in the valid area
        area_height = min(44, h)
        area_width = min(94, w)

        for _ in range(self.num_lines):
            y = random.randint(0, max(0, area_height - self.line_thickness))
            mask[y:y + self.line_thickness, 0:area_width] = 0.0

        return torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


def generate_dataset(
    num_field_pairs=5000,     # Number of div-free field pairs to generate
    masks_per_pair=8,         # Number of different masks per field pair
    save_path=None,
    batch_save_size=1000      # Save every N samples to avoid OOM
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

    # Setup device
    device = get_device()
    print(f"Using device: {device}")

    if save_path is None:
        save_path = BASE_DIR / "results" / "combnet_training_data.pt"

    # Field shape (matching ocean data: 64 height x 128 width)
    channels, height, width = 2, 64, 128
    field_shape = (1, channels, height, width)
    
    # Mask generators - variety of widths and positions
    mask_generators = [
        SimpleStraightLineMaskGenerator(1, 1),
        SimpleStraightLineMaskGenerator(2, 1),
        SimpleStraightLineMaskGenerator(3, 1),
        SimpleStraightLineMaskGenerator(4, 1),
        SimpleStraightLineMaskGenerator(5, 1),
    ]

    # Storage (accumulate in batches to avoid OOM)
    all_datasets = []
    current_batch = {
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
    print(f"  Batch save size: {batch_save_size}")
    print()

    sample_count = 0

    for pair_idx in tqdm(range(num_field_pairs), desc="Generating field pairs"):
        # Use fixed field shape (no ocean data needed!)
        actual_shape = field_shape

        # Generate two independent div-free noise fields (once per pair)
        field_a = generate_divergence_free_noise(actual_shape, device)
        field_b = generate_divergence_free_noise(actual_shape, device)

        # Apply multiple different masks to this field pair
        for mask_idx in range(masks_per_pair):
            # Random mask generator (different position each time)
            mask_gen = np.random.choice(mask_generators)
            mask = mask_gen.generate_mask(actual_shape).to(device)

            # Skip if mask doesn't cover enough area
            if mask.sum() < 10:
                continue

            # Naive merge (this has a boundary discontinuity)
            naive = field_a * (1 - mask) + field_b * mask

            # Store
            current_batch['known'].append(field_a.cpu())
            current_batch['inpainted'].append(field_b.cpu())
            current_batch['mask'].append(mask.cpu())
            current_batch['naive'].append(naive.cpu())

            sample_count += 1

            # Save batch and clear memory when we hit batch size
            if len(current_batch['known']) >= batch_save_size:
                batch_dataset = {
                    'known': torch.cat(current_batch['known'], dim=0),
                    'inpainted': torch.cat(current_batch['inpainted'], dim=0),
                    'mask': torch.cat(current_batch['mask'], dim=0),
                    'naive': torch.cat(current_batch['naive'], dim=0),
                }
                all_datasets.append(batch_dataset)

                # Clear current batch to free memory
                current_batch = {'known': [], 'inpainted': [], 'mask': [], 'naive': []}

                # Clear GPU cache
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

    # Handle remaining samples
    if len(current_batch['known']) > 0:
        batch_dataset = {
            'known': torch.cat(current_batch['known'], dim=0),
            'inpainted': torch.cat(current_batch['inpainted'], dim=0),
            'mask': torch.cat(current_batch['mask'], dim=0),
            'naive': torch.cat(current_batch['naive'], dim=0),
        }
        all_datasets.append(batch_dataset)

    # Combine all batches
    print(f"\nCollected {sample_count} samples in {len(all_datasets)} batches")
    print("Combining batches...")

    final_dataset = {
        'known': torch.cat([d['known'] for d in all_datasets], dim=0),
        'inpainted': torch.cat([d['inpainted'] for d in all_datasets], dim=0),
        'mask': torch.cat([d['mask'] for d in all_datasets], dim=0),
        'naive': torch.cat([d['naive'] for d in all_datasets], dim=0),
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
        num_field_pairs=2500,   # 2500 unique div-free field pairs
        masks_per_pair=8,       # 8 different masks per pair
        batch_save_size=1000,   # Save in batches to avoid OOM
        # Total: ~20,000 samples (reduced for Colab free tier)
    )
