#!/usr/bin/env python3
"""
Inference comparison: Topology-aware DDPM vs Baseline DDPM (both standard_attn).

Both models are unconditional 2-channel MyUNet_Attn, eps-prediction, T=250.
- Baseline: Exp 08 repaint_gaussian_attn (MSE loss only)
- Topo-aware: Exp 10 topo_aware_uncond_eps (MSE + vorticity/divergence penalty)

Uses RePaint inpainting with the same masks/data for fair comparison.
Saves results as .pt, then prints a comparison table.
"""

import sys
import time
from pathlib import Path

import torch
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from data_prep.data_initializer import DDInitializer
from ddpm.neural_networks.ddpm import GaussianDDPM
from ddpm.neural_networks.unets.unet_xl_attn import MyUNet_Attn
from ddpm.utils.inpainting_utils import repaint_standard
from ddpm.helper_functions.masks.n_coverage_mask import CoverageMaskGenerator

# ── Config ──────────────────────────────────────────────────────────
N_STEPS = 250
N_SAMPLES = 10          # number of validation samples to evaluate
RESAMPLE_STEPS = 5      # RePaint resample iterations per timestep
MASK_COVERAGE = 0.90    # 90% missing (standard test regime)
SEED = 42

# Model paths
TOPO_WEIGHTS = BASE_DIR / "experiments/10_topology_metrics/topo_aware_training/results/inpaint_gaussian_t250_best_ema_weights.pt"
BASELINE_WEIGHTS = BASE_DIR / "experiments/08_network_architecture/repaint_gaussian_attn/results/inpaint_gaussian_t250_best_weights.pt"

OUT_PT = BASE_DIR / "experiments/10_topology_metrics/topo_aware_training/results/topo_vs_baseline_inference.pt"

# ── Device ──────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Device: {device}")

# ── Data ────────────────────────────────────────────────────────────
dd = DDInitializer()
standardizer = dd.get_standardizer()
noise_strategy = dd.get_noise_strategy()
val_data = dd.get_validation_data()

print(f"Validation samples: {len(val_data)}")


# ── Load model ──────────────────────────────────────────────────────
def load_model(weight_path, device):
    """Load a standard_attn DDPM from state_dict weights."""
    unet = MyUNet_Attn(n_steps=N_STEPS)
    ddpm = GaussianDDPM(unet, n_steps=N_STEPS, min_beta=0.0001, max_beta=0.02,
                        device=device, image_chw=(2, 64, 128))
    state = torch.load(weight_path, map_location="cpu", weights_only=False)
    # Handle both full checkpoint and plain state_dict
    if isinstance(state, dict) and "model_state_dict" in state:
        ddpm.load_state_dict(state["model_state_dict"])
    else:
        ddpm.load_state_dict(state)
    ddpm = ddpm.to(device)
    ddpm.eval()
    return ddpm


print("\nLoading topology-aware model (EMA)...")
topo_ddpm = load_model(TOPO_WEIGHTS, device)
print("Loading baseline model (Exp 08)...")
baseline_ddpm = load_model(BASELINE_WEIGHTS, device)


# ── Mask generator ──────────────────────────────────────────────────
# CoverageMaskGenerator(coverage_ratio) — ratio is the fraction of ocean
# cells that the BFS robot has OBSERVED (known). After inversion in the
# class, returned mask has 1 = missing, 0 = known.
# For 90% missing → coverage_ratio = 0.10 (10% known/observed).
OBSERVED_FRACTION = 1.0 - MASK_COVERAGE
mask_gen = CoverageMaskGenerator(coverage_ratio=OBSERVED_FRACTION)


# ── Run inference ───────────────────────────────────────────────────
def compute_metrics(pred_std, gt_std, mask):
    """Compute MSE and MAE in standardized space over masked region."""
    diff = (pred_std - gt_std) * mask
    n = mask.sum().clamp(min=1)
    mse = (diff ** 2).sum() / n
    mae = diff.abs().sum() / n
    return mse.item(), mae.item()


def compute_divergence(field_std, standardizer):
    """Compute mean |divergence| of the physical velocity field in ocean region."""
    # Unstandardize to physical units
    field_phys = standardizer.unstandardize(field_std.squeeze(0)).unsqueeze(0)
    u = field_phys[:, 0, :, :]  # (1, H, W)
    v = field_phys[:, 1, :, :]

    # Central differences (interior only)
    du_dx = (u[:, :, 2:] - u[:, :, :-2]) / 2.0
    dv_dy = (v[:, 2:, :] - v[:, :-2, :]) / 2.0

    # Align shapes (interior)
    du_dx_interior = du_dx[:, 1:-1, :]
    dv_dy_interior = dv_dy[:, :, 1:-1]

    div = du_dx_interior + dv_dy_interior

    # Only ocean region (nonzero velocity)
    speed = (u[:, 1:-1, 1:-1] ** 2 + v[:, 1:-1, 1:-1] ** 2).sqrt()
    ocean = (speed > 1e-6).float()
    n_ocean = ocean.sum().clamp(min=1)

    mean_abs_div = (div.abs() * ocean).sum() / n_ocean
    return mean_abs_div.item()


def compute_vorticity_mse(pred_std, gt_std, mask, standardizer):
    """Compute MSE of vorticity (curl) in the masked region."""
    pred_phys = standardizer.unstandardize(pred_std.squeeze(0)).unsqueeze(0)
    gt_phys = standardizer.unstandardize(gt_std.squeeze(0)).unsqueeze(0)

    def curl(field):
        u, v = field[:, 0], field[:, 1]
        dv_dx = (v[:, :, 2:] - v[:, :, :-2]) / 2.0
        du_dy = (u[:, 2:, :] - u[:, :-2, :]) / 2.0
        return dv_dx[:, 1:-1, :] - du_dy[:, :, 1:-1]

    vort_pred = curl(pred_phys)
    vort_gt = curl(gt_phys)

    # Mask in interior
    mask_1ch = mask[:, 0:1, 1:-1, 1:-1]
    diff = (vort_pred.unsqueeze(1) - vort_gt.unsqueeze(1)) * mask_1ch
    n = mask_1ch.sum().clamp(min=1)
    return (diff ** 2).sum().item() / n.item()


print(f"\n{'='*80}")
print(f"TOPO-AWARE vs BASELINE DDPM INFERENCE ({N_SAMPLES} samples, {MASK_COVERAGE*100:.0f}% missing)")
print(f"{'='*80}")

torch.manual_seed(SEED)
np.random.seed(SEED)

results = {
    'topo_mse': [], 'topo_mae': [], 'topo_div': [], 'topo_vort_mse': [],
    'base_mse': [], 'base_mae': [], 'base_div': [], 'base_vort_mse': [],
    'gt_div': [],
    'preds_topo': [], 'preds_base': [],
    'gt_fields': [], 'masks': [],
    'val_indices': [],
}

for i in range(N_SAMPLES):
    vi = i  # sequential validation indices
    input_image = val_data[vi][0].unsqueeze(0)  # (1, 2, 64, 128) standardized

    # Generate mask
    mask_1ch = mask_gen.generate_mask(input_image.shape).to(device)
    mask = mask_1ch.expand(-1, 2, -1, -1)  # (1, 2, 64, 128)

    gt_std = input_image.to(device)

    print(f"\n[Sample {i+1}/{N_SAMPLES}] val_idx={vi}, mask coverage={mask.mean():.2%}")

    # ── Baseline RePaint ──
    t0 = time.time()
    base_out = repaint_standard(
        baseline_ddpm, gt_std, mask,
        n_samples=1, device=device,
        noise_strategy=noise_strategy,
        prediction_target="eps",
        resample_steps=RESAMPLE_STEPS,
    )
    t_base = time.time() - t0

    base_mse, base_mae = compute_metrics(base_out, gt_std, mask)
    base_div = compute_divergence(base_out, standardizer)
    base_vort = compute_vorticity_mse(base_out, gt_std, mask, standardizer)

    # ── Topo-aware RePaint ──
    # Reset RNG to same state for fair comparison
    torch.manual_seed(SEED + i * 1000)
    t0 = time.time()
    topo_out = repaint_standard(
        topo_ddpm, gt_std, mask,
        n_samples=1, device=device,
        noise_strategy=noise_strategy,
        prediction_target="eps",
        resample_steps=RESAMPLE_STEPS,
    )
    t_topo = time.time() - t0

    topo_mse, topo_mae = compute_metrics(topo_out, gt_std, mask)
    topo_div = compute_divergence(topo_out, standardizer)
    topo_vort = compute_vorticity_mse(topo_out, gt_std, mask, standardizer)

    gt_div = compute_divergence(gt_std, standardizer)

    print(f"  Baseline: MSE={base_mse:.6f}  MAE={base_mae:.4f}  |div|={base_div:.6f}  vort_MSE={base_vort:.8f}  ({t_base:.1f}s)")
    print(f"  Topo:     MSE={topo_mse:.6f}  MAE={topo_mae:.4f}  |div|={topo_div:.6f}  vort_MSE={topo_vort:.8f}  ({t_topo:.1f}s)")
    print(f"  GT div:   {gt_div:.6f}")

    results['base_mse'].append(base_mse)
    results['base_mae'].append(base_mae)
    results['base_div'].append(base_div)
    results['base_vort_mse'].append(base_vort)
    results['topo_mse'].append(topo_mse)
    results['topo_mae'].append(topo_mae)
    results['topo_div'].append(topo_div)
    results['topo_vort_mse'].append(topo_vort)
    results['gt_div'].append(gt_div)
    results['preds_base'].append(base_out.cpu())
    results['preds_topo'].append(topo_out.cpu())
    results['gt_fields'].append(gt_std.cpu())
    results['masks'].append(mask.cpu())
    results['val_indices'].append(vi)

# ── Save .pt ────────────────────────────────────────────────────────
save_data = {k: (torch.stack(v) if isinstance(v[0], torch.Tensor) else v)
             for k, v in results.items()}
torch.save(save_data, OUT_PT)
print(f"\nSaved results to {OUT_PT}")

# ── Summary table ───────────────────────────────────────────────────
print(f"\n{'='*80}")
print(f"SUMMARY ({N_SAMPLES} samples, {MASK_COVERAGE*100:.0f}% missing, resample={RESAMPLE_STEPS})")
print(f"{'='*80}")

b_mse = np.mean(results['base_mse'])
t_mse = np.mean(results['topo_mse'])
b_mae = np.mean(results['base_mae'])
t_mae = np.mean(results['topo_mae'])
b_div = np.mean(results['base_div'])
t_div = np.mean(results['topo_div'])
b_vort = np.mean(results['base_vort_mse'])
t_vort = np.mean(results['topo_vort_mse'])
gt_div_mean = np.mean(results['gt_div'])

print(f"\n{'Metric':<20} | {'Baseline':>12} | {'Topo-aware':>12} | {'Change':>10} | {'Better?':>8}")
print("-" * 75)
print(f"{'MSE':<20} | {b_mse:>12.6f} | {t_mse:>12.6f} | {(t_mse/b_mse - 1)*100:>+9.1f}% | {'TOPO' if t_mse < b_mse else 'BASE':>8}")
print(f"{'MAE':<20} | {b_mae:>12.6f} | {t_mae:>12.6f} | {(t_mae/b_mae - 1)*100:>+9.1f}% | {'TOPO' if t_mae < b_mae else 'BASE':>8}")
print(f"{'Mean |div|':<20} | {b_div:>12.6f} | {t_div:>12.6f} | {(t_div/b_div - 1)*100:>+9.1f}% | {'TOPO' if t_div < b_div else 'BASE':>8}")
print(f"{'Vorticity MSE':<20} | {b_vort:>12.8f} | {t_vort:>12.8f} | {(t_vort/b_vort - 1)*100:>+9.1f}% | {'TOPO' if t_vort < b_vort else 'BASE':>8}")
print(f"{'GT |div| (ref)':<20} | {gt_div_mean:>12.6f} |")

# Per-sample wins
mse_wins = sum(1 for t, b in zip(results['topo_mse'], results['base_mse']) if t < b)
div_wins = sum(1 for t, b in zip(results['topo_div'], results['base_div']) if t < b)
vort_wins = sum(1 for t, b in zip(results['topo_vort_mse'], results['base_vort_mse']) if t < b)

print(f"\nPer-sample wins (topo/total): MSE={mse_wins}/{N_SAMPLES}  |div|={div_wins}/{N_SAMPLES}  vort={vort_wins}/{N_SAMPLES}")

print("\nDone.")
