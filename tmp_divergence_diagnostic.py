"""
Diagnostic: How much divergence does our ocean velocity dataset have?
Quantifies irrotational vs solenoidal energy via Helmholtz-Hodge decomposition.
"""
import numpy as np
import torch
import sys
sys.path.insert(0, ".")
from scipy.io import loadmat
from ddpm.helper_functions.HH_decomp import decompose_vector_field
from ddpm.helper_functions.compute_divergence import compute_divergence

# Load raw data from .mat file
mat_data = loadmat("data/rams_head/stjohn_hourly_5m_velocity_ramhead_v2.mat")
u_raw = mat_data['u']  # (94, 44, T)
v_raw = mat_data['v']  # (94, 44, T)

# Stack into (94, 44, 2, T) then -> (T, 2, 44, 94)
all_raw = np.stack([u_raw, v_raw], axis=2)
all_data = np.transpose(all_raw, (3, 2, 1, 0)).astype(np.float32)

# Build ocean mask from first sample
sample0 = all_data[0]
ocean_mask = ~np.isnan(sample0[0])  # (44, 94)
n_ocean = ocean_mask.sum()

# Replace NaN with 0 for computation
all_data = np.nan_to_num(all_data, nan=0.0)

N = all_data.shape[0]
print(f"Dataset: {N} samples, shape per sample: (2, 44, 94)")
print(f"Ocean pixels: {n_ocean} / {44*94} ({100*n_ocean/(44*94):.1f}%)")
print()

# Per-sample statistics
energy_total = []
energy_sol = []
energy_irr = []
div_rms_list = []
vort_rms_list = []
irr_frac_list = []

mask_t = torch.from_numpy(ocean_mask.astype(np.float32))

for i in range(N):
    u = all_data[i, 0]  # (44, 94)
    v = all_data[i, 1]
    
    # Compute divergence (central differences)
    div = compute_divergence(
        torch.from_numpy(u), torch.from_numpy(v)
    )
    if isinstance(div, torch.Tensor):
        div = div.numpy()
    
    # Compute vorticity
    # ω = dv/dx - du/dy (central diff)
    dvdx = np.zeros_like(v)
    dudy = np.zeros_like(u)
    dvdx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / 2.0
    dudy[1:-1, :] = (u[2:, :] - u[:-2, :]) / 2.0
    vort = dvdx - dudy
    
    # Helmholtz decomposition
    field = torch.from_numpy(np.stack([u, v], axis=-1))  # (44, 94, 2)
    (_, _), (u_irr, v_irr), (u_sol, v_sol) = decompose_vector_field(field)
    
    u_irr = u_irr.numpy()
    v_irr = v_irr.numpy()
    u_sol = u_sol.numpy()
    v_sol = v_sol.numpy()
    
    # Energy over ocean pixels only
    mask = ocean_mask.astype(bool)
    E_total = 0.5 * np.mean(u[mask]**2 + v[mask]**2)
    E_sol = 0.5 * np.mean(u_sol[mask]**2 + v_sol[mask]**2)
    E_irr = 0.5 * np.mean(u_irr[mask]**2 + v_irr[mask]**2)
    
    energy_total.append(E_total)
    energy_sol.append(E_sol)
    energy_irr.append(E_irr)
    irr_frac_list.append(E_irr / E_total if E_total > 0 else 0)
    
    div_rms = np.sqrt(np.mean(div[mask]**2))
    vort_rms = np.sqrt(np.mean(vort[mask]**2))
    div_rms_list.append(div_rms)
    vort_rms_list.append(vort_rms)

energy_total = np.array(energy_total)
energy_sol = np.array(energy_sol)
energy_irr = np.array(energy_irr)
irr_frac = np.array(irr_frac_list)
div_rms = np.array(div_rms_list)
vort_rms = np.array(vort_rms_list)

print("=" * 60)
print("HELMHOLTZ-HODGE ENERGY DECOMPOSITION")
print("=" * 60)
print(f"Total KE (mean ± std):       {energy_total.mean():.6f} ± {energy_total.std():.6f}")
print(f"Solenoidal KE (mean ± std):  {energy_sol.mean():.6f} ± {energy_sol.std():.6f}")
print(f"Irrotational KE (mean ± std):{energy_irr.mean():.6f} ± {energy_irr.std():.6f}")
print()
print(f"Irrotational fraction of total KE:")
print(f"  Mean:   {100*irr_frac.mean():.2f}%")
print(f"  Median: {100*np.median(irr_frac):.2f}%")
print(f"  Min:    {100*irr_frac.min():.2f}%")
print(f"  Max:    {100*irr_frac.max():.2f}%")
print(f"  P90:    {100*np.percentile(irr_frac, 90):.2f}%")
print()
print("=" * 60)
print("DIVERGENCE vs VORTICITY (RMS over ocean)")
print("=" * 60)
print(f"Divergence RMS (mean ± std):  {div_rms.mean():.6f} ± {div_rms.std():.6f}")
print(f"Vorticity RMS (mean ± std):   {vort_rms.mean():.6f} ± {vort_rms.std():.6f}")
print(f"Ratio div/vort RMS (mean):    {(div_rms/vort_rms).mean():.4f}")
print(f"  → divergence is {100*(div_rms/vort_rms).mean():.1f}% of vorticity magnitude")
print()

# Speed statistics for context
speeds = np.sqrt(all_data[:, 0]**2 + all_data[:, 1]**2)
mask_3d = np.broadcast_to(ocean_mask, speeds.shape)
ocean_speeds = speeds[mask_3d.astype(bool)]
print("=" * 60)
print("VELOCITY MAGNITUDE CONTEXT")
print("=" * 60)
print(f"Mean speed (ocean):  {ocean_speeds.mean():.4f} m/s")
print(f"Std speed:           {ocean_speeds.std():.4f} m/s")
print(f"Max speed:           {ocean_speeds.max():.4f} m/s")
print()

# Reconstruction error if we only kept solenoidal component
recon_errors = []
for i in range(N):
    u = all_data[i, 0]
    v = all_data[i, 1]
    field = torch.from_numpy(np.stack([u, v], axis=-1))
    (_, _), (_, _), (u_sol, v_sol) = decompose_vector_field(field)
    u_sol = u_sol.numpy()
    v_sol = v_sol.numpy()
    mask = ocean_mask.astype(bool)
    mse = np.mean((u[mask] - u_sol[mask])**2 + (v[mask] - v_sol[mask])**2)
    recon_errors.append(mse)

recon_errors = np.array(recon_errors)
print("=" * 60)
print("STREAMFUNCTION RECONSTRUCTION ERROR")
print("(if we ONLY kept the div-free component)")
print("=" * 60)
print(f"MSE (mean ± std):  {recon_errors.mean():.6f} ± {recon_errors.std():.6f}")
print(f"RMSE (mean):       {np.sqrt(recon_errors.mean()):.6f}")
print(f"As % of total KE:  {100*recon_errors.mean()/(2*energy_total.mean()):.2f}%")
print(f"  → This is the irreducible error floor for Proposal 1")
