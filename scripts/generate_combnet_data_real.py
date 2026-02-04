#!/usr/bin/env python
"""
Generate training dataset for CombNet using REAL DDPM inpainting outputs.

This creates training pairs that match the actual boundary discontinuities
the CombNet will see during inference - small, realistic mismatches rather
than completely independent random fields.

The key insight: during DDPM backward process, both the known region and 
inpainted region are trying to reconstruct the SAME underlying flow field,
so discontinuities are subtle prediction errors, not random differences.

Usage:
    python scripts/generate_combnet_data_real.py
    
Output:
    results/combnet_training_data_real.pt
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
from ddpm.neural_networks.ddpm import GaussianDDPM
from ddpm.neural_networks.unets.unet_xl import MyUNet
from ddpm.helper_functions.masks.straigth_line import StraightLineMaskGenerator
from ddpm.helper_functions.masks.n_coverage_mask import CoverageMaskGenerator
from ddpm.helper_functions.masks.random_path import RandomPathMaskGenerator


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_trained_ddpm(model_path, device, dd):
    """Load a trained DDPM model."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    n_steps = checkpoint.get('n_steps', dd.get_attribute("noise_steps"))
    noise_strategy = checkpoint.get('noise_strategy', dd.get_noise_strategy())
    
    # Create and load model
    network = MyUNet(n_steps)
    ddpm = GaussianDDPM(network, n_steps=n_steps, device=device)
    ddpm.load_state_dict(checkpoint['model_state_dict'])
    ddpm.eval()
    
    return ddpm, noise_strategy, n_steps


def run_partial_backward_process(ddpm, noised_images, input_image, mask, noise_strategy, device, 
                                  capture_timesteps=None):
    """
    Run the DDPM backward process and capture intermediate states.
    
    This mimics what happens during inpainting but captures the raw
    known/inpainted fields BEFORE the CombNet would fix them.
    
    Args:
        ddpm: Trained DDPM model
        noised_images: Pre-computed forward noised images (shared across masks)
        input_image: Original clean image [1, 2, H, W]
        mask: Binary mask [1, 2, H, W], 1=unknown, 0=known
        noise_strategy: Noise generation strategy
        device: torch device
        capture_timesteps: List of timesteps to capture (default: final few)
    
    Returns:
        List of (known, inpainted, naive, mask) tuples at each captured timestep
    """
    n_steps = ddpm.n_steps
    
    if capture_timesteps is None:
        # Capture across the full range
        capture_timesteps = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
    
    # Initialize with noise in unknown region
    noise = noise_strategy(input_image, torch.tensor([n_steps], device=device))
    x = noised_images[n_steps] * (1 - mask) + (noise * mask)
    
    captured_samples = []
    
    # Run backward process
    for t in range(n_steps - 1, -1, -1):
        # Denoise one step
        time_tensor = torch.full((1, 1), t, device=device, dtype=torch.long)
        epsilon_theta = ddpm.backward(x, time_tensor)
        
        alpha_t = ddpm.alphas[t].to(device)
        alpha_t_bar = ddpm.alpha_bars[t].to(device)
        
        if noise_strategy.get_gaussian_scaling():
            inpainted = (1 / alpha_t.sqrt()) * (
                x - ((1 - alpha_t) / (1 - alpha_t_bar).sqrt()) * epsilon_theta
            )
        else:
            inpainted = (1 / alpha_t.sqrt()) * (x - epsilon_theta)
        
        # Add noise for next step (except at t=0)
        if t > 0:
            z = noise_strategy(x, torch.tensor([t], device=device))
            beta_t = ddpm.betas[t].to(device)
            sigma_t = beta_t.sqrt()
            inpainted = inpainted + sigma_t * z
        
        # Known region at this noise level
        known = noised_images[t]
        
        # Naive combination (this has boundary discontinuity)
        naive = known * (1 - mask) + inpainted * mask
        
        # Capture at specified timesteps
        if t in capture_timesteps:
            captured_samples.append({
                'known': known.cpu().clone(),
                'inpainted': inpainted.cpu().clone(),
                'naive': naive.cpu().clone(),
                'mask': mask.cpu().clone(),
                'timestep': t,
            })
        
        # Update x for next iteration (without CombNet - just naive merge)
        x = naive
    
    return captured_samples


def forward_noise_image(ddpm, input_image, noise_strategy, device):
    """Pre-compute forward noising for an image (shared across masks)."""
    n_steps = ddpm.n_steps
    noised_images = [None] * (n_steps + 1)
    noised_images[0] = input_image.clone().to(device)
    
    for t in range(n_steps):
        epsilon = noise_strategy(noised_images[t], torch.tensor([t], device=device))
        noised_images[t + 1] = ddpm(noised_images[t], torch.tensor([t], device=device), epsilon, one_step=True)
    
    return noised_images


def generate_dataset(
    model_path=None,
    config_path=None,
    num_images=200,           # Number of input images to process
    masks_per_image=5,        # Number of different masks per image
    timesteps_per_mask=5,     # Number of timesteps to capture per mask
    save_path=None,
):
    """
    Generate CombNet training data using real DDPM backward process outputs.
    
    This captures the actual boundary discontinuities that occur when
    combining known regions with DDPM predictions during inpainting.
    """
    print("=" * 60)
    print("GENERATING COMBNET TRAINING DATA (REAL DDPM OUTPUTS)")
    print("=" * 60)
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Initialize data
    if config_path:
        dd = DDInitializer(config_path=config_path)
    else:
        dd = DDInitializer()
    
    # Default model path
    if model_path is None:
        model_path = BASE_DIR / "ddpm" / "training" / "training_output" / "div_free_unified_zscore" / "ddpm_ocean_model_best_checkpoint.pt"
    
    if save_path is None:
        save_path = BASE_DIR / "results" / "combnet_training_data_real.pt"
    
    print(f"Model: {model_path}")
    print(f"Output: {save_path}")
    
    # Load DDPM model
    print("\nLoading DDPM model...")
    ddpm, noise_strategy, n_steps = load_trained_ddpm(model_path, device, dd)
    print(f"  Loaded model with {n_steps} diffusion steps")
    
    # Load dataset
    print("\nLoading dataset...")
    train_data = dd.get_training_data()
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    print(f"  {len(train_data)} training samples available")
    
    # Mask generators with various configurations
    mask_generators = [
        StraightLineMaskGenerator(num_lines=1, line_thickness=3),
        StraightLineMaskGenerator(num_lines=2, line_thickness=2),
        StraightLineMaskGenerator(num_lines=3, line_thickness=2),
        CoverageMaskGenerator(coverage_ratio=0.1),
        CoverageMaskGenerator(coverage_ratio=0.2),
        CoverageMaskGenerator(coverage_ratio=0.3),
        RandomPathMaskGenerator(num_lines=2, line_length=30, line_thickness=3),
        RandomPathMaskGenerator(num_lines=3, line_length=20, line_thickness=2),
    ]
    
    # Timesteps to capture across the FULL range of diffusion
    # CombNet is applied at every step, so it needs to handle all noise levels
    capture_timesteps = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
    
    # Incremental saving: save every N images to avoid memory issues
    save_every = 50  # Save checkpoint every 50 images
    
    # Storage (will be cleared after each incremental save)
    all_samples = {
        'known': [],
        'inpainted': [],
        'naive': [],
        'mask': [],
        'timestep': [],
    }
    
    total_samples = num_images * masks_per_image * len(capture_timesteps)
    print(f"\nGenerating data:")
    print(f"  Images to process: {num_images}")
    print(f"  Masks per image: {masks_per_image}")
    print(f"  Timesteps per mask: {len(capture_timesteps)}")
    print(f"  Expected total samples: ~{total_samples}")
    print(f"  Incremental save every: {save_every} images")
    print()
    
    sample_count = 0
    chunk_count = 0
    # Use /tmp for chunks to avoid iCloud sync issues
    temp_dir = Path("/tmp/combnet_chunks")
    temp_dir.mkdir(exist_ok=True)
    print(f"  Temp chunk directory: {temp_dir}")
    
    with torch.no_grad():
        for img_idx, batch in enumerate(tqdm(train_loader, total=num_images, desc="Processing images")):
            if img_idx >= num_images:
                break
            
            input_image = batch[0].to(device)  # [1, 2, H, W]
            
            # Apply land mask
            land_mask = (input_image.abs() > 1e-5).float()
            
            # Pre-compute forward noising ONCE per image (shared across all masks)
            noised_images = forward_noise_image(ddpm, input_image, noise_strategy, device)
            
            for mask_idx in range(masks_per_image):
                # Generate random mask
                mask_gen = np.random.choice(mask_generators)
                try:
                    raw_mask = mask_gen.generate_mask(input_image.shape).to(device)
                    mask = raw_mask * land_mask
                except Exception as e:
                    # Some mask generators may fail on certain shapes
                    continue
                
                # Skip if mask doesn't cover enough area
                if mask.sum() < 50 or mask.sum() > mask.numel() * 0.8:
                    continue
                
                # Run partial backward process and capture samples
                try:
                    samples = run_partial_backward_process(
                        ddpm, noised_images, input_image, mask, noise_strategy, device,
                        capture_timesteps=capture_timesteps
                    )
                except Exception as e:
                    print(f"  Warning: Failed to process sample: {e}")
                    continue
                
                # Store samples
                for sample in samples:
                    all_samples['known'].append(sample['known'])
                    all_samples['inpainted'].append(sample['inpainted'])
                    all_samples['naive'].append(sample['naive'])
                    all_samples['mask'].append(sample['mask'])
                    all_samples['timestep'].append(sample['timestep'])
                    sample_count += 1
            
            # Incremental save to avoid OOM
            if (img_idx + 1) % save_every == 0 and len(all_samples['known']) > 0:
                chunk_path = temp_dir / f"chunk_{chunk_count:04d}.pt"
                chunk_data = {
                    'known': torch.cat(all_samples['known'], dim=0),
                    'inpainted': torch.cat(all_samples['inpainted'], dim=0),
                    'naive': torch.cat(all_samples['naive'], dim=0),
                    'mask': torch.cat(all_samples['mask'], dim=0),
                    'timestep': torch.tensor(all_samples['timestep']),
                }
                torch.save(chunk_data, chunk_path)
                print(f"\n  💾 Saved chunk {chunk_count} with {len(all_samples['known'])} samples")
                chunk_count += 1
                # Clear memory
                all_samples = {k: [] for k in all_samples.keys()}
                import gc
                gc.collect()
    
    # Save any remaining samples
    if len(all_samples['known']) > 0:
        chunk_path = temp_dir / f"chunk_{chunk_count:04d}.pt"
        chunk_data = {
            'known': torch.cat(all_samples['known'], dim=0),
            'inpainted': torch.cat(all_samples['inpainted'], dim=0),
            'naive': torch.cat(all_samples['naive'], dim=0),
            'mask': torch.cat(all_samples['mask'], dim=0),
            'timestep': torch.tensor(all_samples['timestep']),
        }
        torch.save(chunk_data, chunk_path)
        print(f"\n  💾 Saved final chunk {chunk_count} with {len(all_samples['known'])} samples")
        chunk_count += 1
    
    print(f"\nCollected {sample_count} samples in {chunk_count} chunks")
    
    if sample_count == 0:
        raise RuntimeError("No samples were generated! Check model path and data.")
    
    # Combine all chunks into final dataset
    print("Combining chunks into final dataset...")
    all_known = []
    all_inpainted = []
    all_naive = []
    all_mask = []
    all_timestep = []
    
    for i in range(chunk_count):
        chunk_path = temp_dir / f"chunk_{i:04d}.pt"
        chunk = torch.load(chunk_path, weights_only=False)
        all_known.append(chunk['known'])
        all_inpainted.append(chunk['inpainted'])
        all_naive.append(chunk['naive'])
        all_mask.append(chunk['mask'])
        all_timestep.append(chunk['timestep'])
        # DON'T delete chunks yet - wait until final file is verified
    
    final_dataset = {
        'known': torch.cat(all_known, dim=0),
        'inpainted': torch.cat(all_inpainted, dim=0),
        'naive': torch.cat(all_naive, dim=0),
        'mask': torch.cat(all_mask, dim=0),
        'timestep': torch.cat(all_timestep, dim=0),
    }
    
    # Free memory from individual chunks
    del all_known, all_inpainted, all_naive, all_mask, all_timestep
    import gc
    gc.collect()
    
    print(f"\nDataset shapes:")
    for key, val in final_dataset.items():
        print(f"  {key}: {val.shape}")
    
    # Compute statistics on boundary discontinuities
    print("\nBoundary discontinuity statistics:")
    naive = final_dataset['naive']
    known = final_dataset['known']
    mask = final_dataset['mask']
    
    # Discontinuity = difference between known and inpainted at boundary
    # Approximate by looking at the naive field's gradient at mask edges
    boundary_diffs = []
    for i in range(min(100, len(naive))):
        m = mask[i, 0]  # Get first channel of mask [64, 128]
        k = known[i]    # [2, 64, 128]
        n = naive[i]    # [2, 64, 128]
        # Difference in known region vs naive
        diff = (k - n).abs().mean(dim=0)  # Average over channels -> [64, 128]
        masked_diff = diff[m > 0.5]
        if masked_diff.numel() > 0:
            boundary_diffs.append(masked_diff.mean().item())
    
    if boundary_diffs:
        print(f"  Mean boundary discontinuity: {np.mean(boundary_diffs):.6f}")
        print(f"  Std boundary discontinuity: {np.std(boundary_diffs):.6f}")
        print(f"  Max boundary discontinuity: {np.max(boundary_diffs):.6f}")
    
    # Save to /tmp first to avoid iCloud issues, then copy
    import shutil
    temp_save_path = Path("/tmp") / save_path.name
    print(f"\nSaving to temp location first: {temp_save_path}")
    torch.save(final_dataset, temp_save_path)
    
    # Verify the saved file loads correctly
    print("Verifying saved file...")
    try:
        verify_data = torch.load(temp_save_path, weights_only=False)
        verify_count = len(verify_data['known'])
        del verify_data
        gc.collect()
        print(f"✅ Verification passed: {verify_count} samples")
    except Exception as e:
        print(f"❌ Verification FAILED: {e}")
        print(f"Temp chunks preserved at: {temp_dir}")
        raise RuntimeError(f"File verification failed: {e}")
    
    # Copy to final destination
    save_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Copying to final destination: {save_path}")
    shutil.copy2(temp_save_path, save_path)
    
    # Verify the copied file too
    print("Verifying copied file...")
    try:
        verify_data = torch.load(save_path, weights_only=False)
        verify_count = len(verify_data['known'])
        del verify_data
        gc.collect()
        print(f"✅ Final verification passed: {verify_count} samples")
    except Exception as e:
        print(f"❌ Final verification FAILED: {e}")
        print(f"Temp chunks preserved at: {temp_dir}")
        print(f"Valid temp file at: {temp_save_path}")
        raise RuntimeError(f"Final file verification failed: {e}")
    
    # NOW we can clean up temp files
    print("\nCleaning up temp files...")
    for i in range(chunk_count):
        chunk_path = temp_dir / f"chunk_{i:04d}.pt"
        if chunk_path.exists():
            chunk_path.unlink()
    if temp_dir.exists():
        temp_dir.rmdir()
    if temp_save_path.exists():
        temp_save_path.unlink()
    
    print(f"\n✅ Saved dataset to: {save_path}")
    
    return final_dataset


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Generate CombNet training data from real DDPM outputs")
    parser.add_argument('--model', type=str, default=None, help='Path to trained DDPM model')
    parser.add_argument('--config', type=str, default=None, help='Path to config YAML')
    parser.add_argument('--num_images', type=int, default=200, help='Number of input images to process')
    parser.add_argument('--masks_per_image', type=int, default=5, help='Number of masks per image')
    parser.add_argument('--output', type=str, default=None, help='Output path for dataset')
    args = parser.parse_args()
    
    generate_dataset(
        model_path=Path(args.model) if args.model else None,
        config_path=Path(args.config) if args.config else None,
        num_images=args.num_images,
        masks_per_image=args.masks_per_image,
        save_path=Path(args.output) if args.output else None,
    )
