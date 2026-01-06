"""
Test script to evaluate DDPM inpainting performance using Gaussian noise.
This script loads a trained model and tests its ability to inpaint masked 
regions of ocean velocity fields using standard Gaussian noise.
"""

import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from data_prep.data_initializer import DDInitializer
from ddpm.neural_networks.ddpm import GaussianDDPM
from ddpm.neural_networks.unets.unet_xl import MyUNet
from ddpm.utils.inpainting_utils import inpaint_generate_new_images, calculate_mse, top_left_crop
from ddpm.utils.noise_utils import GaussianNoise
from ddpm.helper_functions.masks.n_coverage_mask import CoverageMaskGenerator
from ddpm.helper_functions.masks.straigth_line import StraightLineMaskGenerator


def load_model(model_path, device, n_steps=100, min_beta=0.0001, max_beta=0.02):
    """Load a trained DDPM model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Get model parameters from checkpoint or use defaults
    n_steps = checkpoint.get('n_steps', n_steps)
    min_beta = checkpoint.get('min_beta', min_beta)
    max_beta = checkpoint.get('max_beta', max_beta)
    
    model = GaussianDDPM(MyUNet(n_steps), n_steps=n_steps, min_beta=min_beta, 
                         max_beta=max_beta, device=device)
    model.load_state_dict(model_state_dict)
    model.eval()
    
    return model, n_steps, min_beta, max_beta


def visualize_results(original, mask, inpainted, save_path=None):
    """Visualize original, mask, and inpainted results side by side."""
    original = original.cpu().squeeze().numpy()
    mask = mask.cpu().squeeze().numpy()
    inpainted = inpainted.cpu().squeeze().numpy()
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # U component (horizontal velocity)
    axes[0, 0].imshow(original[0], cmap='RdBu_r', aspect='auto')
    axes[0, 0].set_title('Original U')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(mask[0], cmap='gray', aspect='auto')
    axes[0, 1].set_title('Mask (white = inpaint)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(inpainted[0], cmap='RdBu_r', aspect='auto')
    axes[0, 2].set_title('Inpainted U')
    axes[0, 2].axis('off')
    
    diff_u = np.abs(original[0] - inpainted[0]) * mask[0]
    axes[0, 3].imshow(diff_u, cmap='hot', aspect='auto')
    axes[0, 3].set_title('|Error| U (masked region)')
    axes[0, 3].axis('off')
    
    # V component (vertical velocity)
    axes[1, 0].imshow(original[1], cmap='RdBu_r', aspect='auto')
    axes[1, 0].set_title('Original V')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(mask[1], cmap='gray', aspect='auto')
    axes[1, 1].set_title('Mask (white = inpaint)')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(inpainted[1], cmap='RdBu_r', aspect='auto')
    axes[1, 2].set_title('Inpainted V')
    axes[1, 2].axis('off')
    
    diff_v = np.abs(original[1] - inpainted[1]) * mask[1]
    axes[1, 3].imshow(diff_v, cmap='hot', aspect='auto')
    axes[1, 3].set_title('|Error| V (masked region)')
    axes[1, 3].axis('off')
    
    plt.suptitle('Gaussian Noise Inpainting Test', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


def visualize_vector_field(original, inpainted, mask, save_path=None, subsample=4):
    """Visualize vector fields with quiver plots."""
    original = original.cpu().squeeze().numpy()
    inpainted = inpainted.cpu().squeeze().numpy()
    mask = mask.cpu().squeeze().numpy()[0]  # Single channel mask
    
    H, W = original.shape[1], original.shape[2]
    y, x = np.mgrid[0:H:subsample, 0:W:subsample]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original vector field
    u_orig = original[0][::subsample, ::subsample]
    v_orig = original[1][::subsample, ::subsample]
    mag_orig = np.sqrt(u_orig**2 + v_orig**2)
    axes[0].quiver(x, y, u_orig, v_orig, mag_orig, cmap='viridis', scale=3)
    axes[0].set_title('Original Vector Field')
    axes[0].set_aspect('equal')
    axes[0].invert_yaxis()
    
    # Inpainted vector field
    u_inp = inpainted[0][::subsample, ::subsample]
    v_inp = inpainted[1][::subsample, ::subsample]
    mag_inp = np.sqrt(u_inp**2 + v_inp**2)
    axes[1].quiver(x, y, u_inp, v_inp, mag_inp, cmap='viridis', scale=3)
    axes[1].set_title('Inpainted Vector Field (Gaussian Noise)')
    axes[1].set_aspect('equal')
    axes[1].invert_yaxis()
    
    # Mask overlay
    axes[2].imshow(mask, cmap='gray', aspect='auto')
    axes[2].set_title('Mask (white = inpainted region)')
    axes[2].axis('off')
    
    plt.suptitle('Vector Field Comparison', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved vector field visualization to {save_path}")
    
    plt.show()


def main():
    print("=" * 60)
    print("Gaussian Noise Inpainting Test")
    print("=" * 60)
    
    # Initialize data
    dd = DDInitializer()
    device = dd.get_device()
    print(f"Using device: {device}")
    
    # Model path - using the weekend model for gaussian noise
    model_path = BASE_DIR / "ddpm/Trained_Models/weekend_ddpm_ocean_model.pt"
    
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        print("Available models:")
        for f in (BASE_DIR / "ddpm/Trained_Models").glob("*.pt"):
            print(f"  - {f.name}")
        return
    
    print(f"Loading model from: {model_path}")
    model, n_steps, min_beta, max_beta = load_model(model_path, device)
    print(f"Model loaded: n_steps={n_steps}, min_beta={min_beta}, max_beta={max_beta}")
    
    # Reinitialize with model parameters
    dd.reinitialize(min_beta, max_beta, n_steps, dd.get_standardizer())
    
    # Load validation data (to match model_inpainter)
    test_loader = DataLoader(dd.get_validation_data(), batch_size=1, shuffle=True)
    
    # Create Gaussian noise strategy
    gaussian_noise = GaussianNoise()
    print(f"Using Gaussian noise strategy")
    
    # Create mask generator - use StraightLineMaskGenerator to match model_inpainter
    mask_generator = StraightLineMaskGenerator(1)
    
    # Get a test sample
    batch = next(iter(test_loader))
    input_image = batch[0].to(device)
    sample_id = batch[1].item()
    print(f"Testing on sample ID: {sample_id}")
    
    # Unstandardize for visualization
    standardizer = dd.get_standardizer()
    input_image_original = standardizer.unstandardize(input_image.squeeze(0)).unsqueeze(0).to(device)
    
    # Generate mask and apply land mask to only inpaint valid ocean regions
    land_mask = (input_image_original.abs() > 1e-5).float().to(device)
    raw_mask = mask_generator.generate_mask(input_image.shape).to(device)
    mask = raw_mask * land_mask
    
    mask_percentage = 100 * mask.sum().item() / mask.numel()
    print(f"Mask coverage: {mask_percentage:.2f}%")
    
    # Perform inpainting with Gaussian noise
    print("Running inpainting with Gaussian noise...")
    print(f"  - n_steps: {model.n_steps}")
    print(f"  - resample_steps: 5")
    print(f"  - use_comb_net: {dd.get_use_comb_net()}")
    resample_steps = 5  # Match model_inpainter config
    
    import time
    start_time = time.time()
    
    with torch.no_grad():
        print("Starting inpaint_generate_new_images...")
        inpainted = inpaint_generate_new_images(
            model, 
            input_image, 
            mask, 
            n_samples=1,
            device=device, 
            resample_steps=resample_steps, 
            noise_strategy=gaussian_noise
        )
    
    elapsed = time.time() - start_time
    print(f"Inpainting completed in {elapsed:.2f} seconds")
    
    # Unstandardize the result
    inpainted = standardizer.unstandardize(inpainted.squeeze(0)).unsqueeze(0).to(device)
    
    # Crop to valid ocean region (as done in original code)
    input_cropped = top_left_crop(input_image_original, 44, 94)
    inpainted_cropped = top_left_crop(inpainted, 44, 94)
    mask_cropped = top_left_crop(mask, 44, 94)
    
    # Calculate MSE
    mse = calculate_mse(input_cropped, inpainted_cropped, mask_cropped, normalize=True)
    print(f"\n{'=' * 40}")
    print(f"Results:")
    print(f"  MSE (normalized, masked region): {mse.item():.6f}")
    print(f"{'=' * 40}")
    
    # Save results
    results_dir = BASE_DIR / "ddpm/Testing/results/gaussian_test"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualize
    visualize_results(
        input_cropped, 
        mask_cropped, 
        inpainted_cropped,
        save_path=results_dir / f"gaussian_inpaint_sample_{sample_id}.png"
    )
    
    visualize_vector_field(
        input_cropped,
        inpainted_cropped,
        mask_cropped,
        save_path=results_dir / f"gaussian_vectors_sample_{sample_id}.png"
    )
    
    # Save tensors for further analysis
    torch.save(input_cropped, results_dir / f"original_{sample_id}.pt")
    torch.save(inpainted_cropped, results_dir / f"inpainted_gaussian_{sample_id}.pt")
    torch.save(mask_cropped, results_dir / f"mask_{sample_id}.pt")
    print(f"\nSaved tensor files to {results_dir}")
    
    print("\nTest complete!")


if __name__ == "__main__":
    main()
