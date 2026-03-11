#!/usr/bin/env python
"""Quick diagnostic: check FiLM model output magnitude."""
import torch, sys
sys.path.insert(0, '.')
from data_prep.data_initializer import DDInitializer
from ddpm.neural_networks.ddpm import GaussianDDPM
from ddpm.neural_networks.unets.unet_film import MyUNet_FiLM
from ddpm.utils.inpainting_utils import mask_aware_inpaint
from ddpm.utils.noise_utils import get_noise_strategy
from ddpm.helper_functions.masks.border_mask import BorderMaskGenerator
import numpy as np

dd = DDInitializer()
device = dd.get_device()
standardizer = dd.get_standardizer()

# Load model
ckpt = torch.load(
    'ddpm/training/training_output/inpaint_film_t250/'
    'inpaint_spectral_div_free_t250_best_checkpoint.pt',
    map_location=device, weights_only=False
)
network = MyUNet_FiLM(n_steps=250, time_emb_dim=100, in_channels=5)
ddpm = GaussianDDPM(network, n_steps=250, min_beta=0.0004, max_beta=0.08, device=device)
ddpm.load_state_dict(ckpt['model_state_dict'])
ddpm = ddpm.to(device)
ddpm.eval()
noise_strategy = get_noise_strategy('spectral_div_free')

# Get one test sample
val_data = dd.get_validation_data()
input_image = val_data[0][0].unsqueeze(0).to(device)
input_orig = standardizer.unstandardize(input_image.squeeze(0)).to(device).unsqueeze(0)

# Mask
h, w = 64, 128
mask = np.ones((h, w), dtype=np.float32)
mask[22:23, 0:94] = 0.0
mask_t = torch.tensor(mask).unsqueeze(0).unsqueeze(0)
border = BorderMaskGenerator().generate_mask(input_image.shape)
mask_t = mask_t.to(border.device) * border
land_mask = (input_orig.abs() > 1e-5).float().to(device)
missing_mask = mask_t.to(device) * land_mask

print(f"Input (std): min={input_image.min():.3f} max={input_image.max():.3f} mean={input_image.mean():.3f}")
print(f"Physical:    min={input_orig.min():.4f} max={input_orig.max():.4f}")
print(f"Missing mask: {missing_mask.sum():.0f} pixels")

# Check initial noise magnitude
torch.manual_seed(42)
x = noise_strategy(torch.zeros(1, 2, h, w, device=device), torch.tensor([249], device=device))
print(f"Initial noise: min={x.min():.3f} max={x.max():.3f} std={x.std():.3f}")

# Check what standard Gaussian noise looks like
g = torch.randn(1, 2, h, w, device=device)
print(f"Gaussian noise: min={g.min():.3f} max={g.max():.3f} std={g.std():.3f}")

# Run mask_aware_inpaint with spectral noise
torch.manual_seed(42)
result_spec = mask_aware_inpaint(
    ddpm, input_image, missing_mask, n_samples=1,
    device=device, noise_strategy=noise_strategy, mask_xt=False,
)
print(f"\nSpectral noise result (std): min={result_spec.min():.3f} max={result_spec.max():.3f} std={result_spec.std():.3f}")
result_phys = standardizer.unstandardize(result_spec.squeeze(0)).to(device).unsqueeze(0)
print(f"Spectral noise result (phys): min={result_phys.min():.4f} max={result_phys.max():.4f}")
mse_spec = ((result_phys - input_orig) * missing_mask).pow(2).sum() / (missing_mask.sum() + 1e-8)
print(f"MSE (spectral): {mse_spec.item():.6f}")

# Now try with Gaussian noise instead
# ── Try starting from intermediate timestep instead of pure noise ──
# Forward-diffuse ground truth to timestep t, then reverse with FiLM conditioning
from tqdm import tqdm

for t_start in [10, 25, 50, 75, 100]:
    torch.manual_seed(42)
    
    # Forward diffuse clean standardized field to t_start
    alpha_bar = ddpm.alpha_bars[t_start].to(device)
    eps_init = noise_strategy(torch.zeros_like(input_image), torch.tensor([t_start], device=device))
    x = alpha_bar.sqrt() * input_image + (1 - alpha_bar).sqrt() * eps_init
    
    # Prep conditioning
    mask_dev = missing_mask.to(device)
    known_mask_dev = 1.0 - mask_dev
    mask_single = mask_dev[:, 0:1]
    known_values = input_image * known_mask_dev
    
    # Reverse from t_start to 0 with conditioning
    ddpm.eval()
    with torch.no_grad():
        for t in range(t_start, -1, -1):
            alpha_t = ddpm.alphas[t].to(device)
            alpha_bar_t = ddpm.alpha_bars[t].to(device)
            beta_t = ddpm.betas[t].to(device)
            tt = torch.full((1, 1), t, device=device, dtype=torch.long)
            
            x_cond = torch.cat([x, mask_single, known_values], dim=1)
            eps_theta = ddpm.network(x_cond, tt)
            
            mu = (1.0 / alpha_t.sqrt()) * (x - ((1 - alpha_t) / (1 - alpha_bar_t).sqrt()) * eps_theta)
            
            if t > 0:
                sigma_t = beta_t.sqrt()
                z = noise_strategy(torch.zeros_like(x), torch.tensor([t], device=device))
                x = mu + sigma_t * z
            else:
                x = mu
    
    result = input_image * known_mask_dev + x * mask_dev
    r_phys = standardizer.unstandardize(result.squeeze(0)).to(device).unsqueeze(0)
    mse = ((r_phys - input_orig) * missing_mask).pow(2).sum() / (missing_mask.sum() + 1e-8)
    print(f"t_start={t_start:3d}: result std={result.std():.3f}, MSE={mse.item():.6f}")

# ── Try GP-init + FiLM reverse ──
print("\nGP-init + FiLM conditional reverse:")
from ddpm.helper_functions.interpolation_tool import gp_fill

gp_out, gp_var = gp_fill(
    input_orig, missing_mask,
    lengthscale=dd.get_attribute("gp_lengthscale"),
    variance=dd.get_attribute("gp_variance"),
    noise=dd.get_attribute("gp_noise"),
    use_double=True,
    kernel_type=dd.get_attribute("gp_kernel_type"),
    coord_system=dd.get_attribute("gp_coord_system"),
    return_variance=True,
)
gp_mse = ((gp_out - input_orig) * missing_mask).pow(2).sum() / (missing_mask.sum() + 1e-8)
print(f"GP baseline MSE: {gp_mse.item():.6f}")

gp_std = standardizer(gp_out.squeeze(0)).to(device).unsqueeze(0)

for t_start in [25, 50, 75]:
    torch.manual_seed(42)
    alpha_bar = ddpm.alpha_bars[t_start].to(device)
    eps_init = noise_strategy(torch.zeros_like(gp_std), torch.tensor([t_start], device=device))
    x = alpha_bar.sqrt() * gp_std + (1 - alpha_bar).sqrt() * eps_init
    
    mask_dev = missing_mask.to(device)
    known_mask_dev = 1.0 - mask_dev
    mask_single = mask_dev[:, 0:1]
    known_values = input_image * known_mask_dev
    
    ddpm.eval()
    with torch.no_grad():
        for t in range(t_start, -1, -1):
            alpha_t = ddpm.alphas[t].to(device)
            alpha_bar_t = ddpm.alpha_bars[t].to(device)
            beta_t = ddpm.betas[t].to(device)
            tt = torch.full((1, 1), t, device=device, dtype=torch.long)
            
            x_cond = torch.cat([x, mask_single, known_values], dim=1)
            eps_theta = ddpm.network(x_cond, tt)
            
            mu = (1.0 / alpha_t.sqrt()) * (x - ((1 - alpha_t) / (1 - alpha_bar_t).sqrt()) * eps_theta)
            
            if t > 0:
                sigma_t = beta_t.sqrt()
                z = noise_strategy(torch.zeros_like(x), torch.tensor([t], device=device))
                x = mu + sigma_t * z
            else:
                x = mu
    
    result = input_image * known_mask_dev + x * mask_dev
    r_phys = standardizer.unstandardize(result.squeeze(0)).to(device).unsqueeze(0)
    mse = ((r_phys - input_orig) * missing_mask).pow(2).sum() / (missing_mask.sum() + 1e-8)
    print(f"GP-init t_start={t_start:3d}: result std={result.std():.3f}, MSE={mse.item():.6f}, ratio vs GP={mse.item()/gp_mse.item():.3f}")
