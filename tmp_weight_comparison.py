#!/usr/bin/env python3
"""Quick A/B/C test: compare different weight files for GP-conditioned model."""
import sys, os, time, torch
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from ddpm.neural_networks.ddpm import GaussianDDPM
from ddpm.neural_networks.unets.unet_film_attn import MyUNet_FiLM_Attn
from ddpm.helper_functions.masks.border_mask import BorderMaskGenerator
from ddpm.helper_functions.standardize_data import ZScoreStandardizer
from ddpm.utils.inpainting_utils import x0_full_reverse_inpaint
from ddpm.utils.noise_utils import get_noise_strategy
from ddpm.helper_functions.interpolation_tool import gp_fill
from data_prep.data_initializer import DDInitializer

# Config
N_STEPS = 250
U_MEAN, U_STD = -0.06929559429949586, 0.1358005549716049
V_MEAN, V_STD = -0.0323937796117541, 0.08899177232117582
standardizer = ZScoreStandardizer(U_MEAN, U_STD, V_MEAN, V_STD)
noise_strategy = get_noise_strategy("gaussian")

RESULTS_DIR = os.path.join(BASE, "experiments/06_gp_forward/gp_conditioned/results")

# Weight files to test
WEIGHT_OPTIONS = {
    "A_ema_early": os.path.join(RESULTS_DIR, "inpaint_gaussian_t250_best_ema_weights.pt"),
    "B_noema_best": os.path.join(RESULTS_DIR, "inpaint_gaussian_t250_best_weights.pt"),
}

# Try to extract EMA from latest checkpoint
LATEST_CKPT = os.path.join(RESULTS_DIR, "inpaint_gaussian_t250_latest.pt")
if os.path.exists(LATEST_CKPT):
    try:
        ckpt = torch.load(LATEST_CKPT, map_location="cpu", weights_only=False)
        print(f"Latest checkpoint keys: {list(ckpt.keys())}")
        if "ema_state" in ckpt:
            print("  -> Has EMA state! Will extract and test.")
            WEIGHT_OPTIONS["C_ema_latest"] = "__ema_from_ckpt__"
        if "epoch" in ckpt:
            print(f"  -> Epoch: {ckpt['epoch']}")
        if "best_test_loss" in ckpt:
            print(f"  -> Best test loss: {ckpt['best_test_loss']:.6f}")
    except Exception as e:
        print(f"Failed to load latest checkpoint: {e}")

# Setup data
dd = DDInitializer()
device = dd.get_device()
val_data = dd.get_validation_data()

# Load GP-Diff reference for comparison
GPDIFF_PT = os.path.join(BASE, "results/eddy_balanced_eval/bulk_eval_eddy_balanced_100.pt")
gpdiff_data = torch.load(GPDIFF_PT, map_location="cpu", weights_only=False)
val_indices = gpdiff_data["val_indices"]

# Use first sample
vi = val_indices[0]
print(f"\nTesting on val sample {vi}")

# Prepare input
dd_std = dd.get_standardizer()
input_image_dd = val_data[vi][0].unsqueeze(0)
input_orig = dd_std.unstandardize(input_image_dd.squeeze(0)).unsqueeze(0).to(device)
input_std = standardizer(input_orig.squeeze(0)).unsqueeze(0).to(device)

# Build mask
land_mask = (input_orig.abs() > 1e-5).float().to(device)
area_h, area_w = 44, 94
raw_mask = torch.ones(1, 1, 64, 128, device=device)
raw_mask[0, 0, area_h // 2, :area_w] = 0.0
border = BorderMaskGenerator().generate_mask(input_std.shape).to(device)
raw_mask = raw_mask * border
missing_mask_1ch = raw_mask * land_mask[:, 0:1]
missing_mask = missing_mask_1ch.expand(-1, 2, -1, -1)

# GP fill
gp_out, gp_var_map = gp_fill(
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
gp_std_field = standardizer(gp_out.squeeze(0)).unsqueeze(0).to(device)

print(f"GP MSE: {gp_mse.item():.6f}")
print(f"\n{'Weight option':<20} {'MSE':>10} {'vs GP':>8} {'Time':>7}")
print("-" * 50)


def test_weights(name, state_dict):
    """Load weights and run full-reverse inference."""
    network = MyUNet_FiLM_Attn(n_steps=N_STEPS, time_emb_dim=256, in_channels=5)
    ddpm = GaussianDDPM(network, n_steps=N_STEPS, min_beta=0.0001, max_beta=0.02, device=device)
    ddpm.load_state_dict(state_dict)
    ddpm = ddpm.to(device)
    ddpm.eval()

    torch.manual_seed(42)
    t0 = time.time()
    with torch.no_grad():
        result_std = x0_full_reverse_inpaint(
            ddpm, input_std, missing_mask,
            n_samples=1, device=device,
            noise_strategy=noise_strategy,
            mask_xt=True,
            known_values_override=gp_std_field,
        )
    elapsed = time.time() - t0

    result_phys = standardizer.unstandardize(result_std.squeeze(0)).to(device).unsqueeze(0)
    mse = ((result_phys - input_orig) * missing_mask).pow(2).sum() / (missing_mask.sum() + 1e-8)
    ratio = mse.item() / gp_mse.item()
    print(f"{name:<20} {mse.item():>10.6f} {ratio:>7.3f}x {elapsed:>6.1f}s")
    return mse.item()


for name, path in WEIGHT_OPTIONS.items():
    if path == "__ema_from_ckpt__":
        # Extract EMA state from checkpoint and apply
        ckpt = torch.load(LATEST_CKPT, map_location="cpu", weights_only=False)
        # The EMA state contains shadow copies of model params
        # We need to apply them to the model
        network = MyUNet_FiLM_Attn(n_steps=N_STEPS, time_emb_dim=256, in_channels=5)
        ddpm = GaussianDDPM(network, n_steps=N_STEPS, min_beta=0.0001, max_beta=0.02, device="cpu")
        ddpm.load_state_dict(ckpt["model_state_dict"])
        # Apply EMA
        from torch_ema import ExponentialMovingAverage
        ema = ExponentialMovingAverage(ddpm.parameters(), decay=0.9999)
        ema.load_state_dict(ckpt["ema_state"])
        ema.apply()  # replace model params with EMA shadow params
        state = ddpm.state_dict()
        ema.restore()
        test_weights(name, state)
    else:
        if not os.path.exists(path):
            print(f"{name:<20} FILE NOT FOUND: {path}")
            continue
        state = torch.load(path, map_location="cpu", weights_only=False)
        test_weights(name, state)
