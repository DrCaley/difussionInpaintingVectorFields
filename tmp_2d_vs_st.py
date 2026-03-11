#!/usr/bin/env python
"""Quick 2D vs ST model comparison on the same test data."""
import torch
import numpy as np
import time

from data_prep.data_initializer import DDInitializer
from data_prep.ocean_sequence_dataset import OceanSequenceDataset
from ddpm.neural_networks.ddpm import GaussianDDPM
from ddpm.neural_networks.unets.unet_xl_attn import MyUNet_Attn
from ddpm.utils.inpainting_utils import repaint_standard
from ddpm.helper_functions.masks.n_coverage_mask import CoverageMaskGenerator

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
dd = DDInitializer()
std = dd.get_standardizer()
ns = dd.get_noise_strategy()

# Load 2D model
ckpt = torch.load(
    "experiments/02_inpaint_algorithm/repaint_gaussian_attn/results/"
    "inpaint_gaussian_t250_best_checkpoint.pt",
    map_location=device, weights_only=False,
)
unet2d = MyUNet_Attn(n_steps=250)
ddpm2d = GaussianDDPM(
    unet2d, n_steps=250, min_beta=0.0001, max_beta=0.02,
    device=device, image_chw=(2, 64, 128),
)
ddpm2d.load_state_dict(ckpt["model_state_dict"])
ddpm2d = ddpm2d.to(device)
ddpm2d.eval()
print(f"2D model loaded: {sum(p.numel() for p in unet2d.parameters()):,} params")

# Load test sequences
test_ds = OceanSequenceDataset(
    data_tensor=dd.test_tensor, n_steps=250,
    noise_strategy=ns, transform=dd.get_transform(), T=13,
)

# Same seed / index as ST test (seed=42)
torch.manual_seed(42)
idx = torch.randperm(len(test_ds))[0].item()
print(f"Test index: {idx}")

x0_seq, _, _ = test_ds[idx]  # (26, 64, 128)

# Mask generation (same as ST test)
mask_gen = CoverageMaskGenerator(0.9)
torch.manual_seed(42)
np.random.seed(42)
ref_shape = (1, 2, 64, 128)
explored = mask_gen.generate_mask(ref_shape).to(device)
missing = 1.0 - explored  # 1=missing, 0=known

# Run 2D model on 3 representative frames: 0, 6, 12
results_2d = {}
for f_idx in [0, 6, 12]:
    x0_frame = x0_seq[f_idx*2:(f_idx+1)*2].unsqueeze(0).to(device)
    
    t0 = time.time()
    pred2d = repaint_standard(
        ddpm2d, x0_frame, missing,
        noise_strategy=ns, resample_steps=3, prediction_target="eps",
    )
    elapsed = time.time() - t0
    
    pred_phys = std.unstandardize(pred2d[0].cpu())
    gt_phys = std.unstandardize(x0_frame[0].cpu())
    mask_cpu = missing[0, :2].cpu() if missing.shape[1] >= 2 else missing[0].expand(2, -1, -1).cpu()
    diff = (pred_phys - gt_phys) * mask_cpu
    mse = (diff ** 2).sum() / mask_cpu.sum().clamp(min=1)
    results_2d[f_idx] = mse.item()
    print(f"  2D frame {f_idx}: MSE = {mse.item():.6f} ({elapsed:.1f}s)")

print()
print("=" * 60)
print("COMPARISON: 2D vs ST (same sample, same mask, 90% coverage)")
print("=" * 60)

# ST results from the first run (seed=42, sample index 627)
st_per_frame = [0.015546, 0.014373, 0.008982, 0.005087, 0.006224,
                0.005944, 0.008905, 0.009526, 0.009862, 0.008326,
                0.006639, 0.004510, 0.003113]

print(f"{'Frame':<8} {'2D MSE':<12} {'ST MSE':<12} {'Ratio ST/2D':<12}")
print("-" * 44)
for f_idx in [0, 6, 12]:
    r2d = results_2d[f_idx]
    rst = st_per_frame[f_idx]
    ratio = rst / r2d if r2d > 0 else float("inf")
    print(f"{f_idx:<8} {r2d:<12.6f} {rst:<12.6f} {ratio:<12.2f}")

avg_2d = np.mean(list(results_2d.values()))
avg_st = np.mean(st_per_frame)
print(f"\n2D avg (3 frames): {avg_2d:.6f}")
print(f"ST avg (all 13):   {avg_st:.6f}")
