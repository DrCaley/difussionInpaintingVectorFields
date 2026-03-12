#!/usr/bin/env python3
"""Evaluate Helmholtz split-decoder UNet — single-step + reverse chain.

Compares against V-CNN baseline and reports MSE, divergence, and
Helmholtz decomposition diagnostics at various coverage levels.

Saves results as .pt first, then generates a summary plot.

Usage:
    PYTHONPATH=. python scripts/eval_helmholtz_split.py
    PYTHONPATH=. python scripts/eval_helmholtz_split.py --n-samples 20 --coverage 0.5 1.0 2.0 5.0
    PYTHONPATH=. python scripts/eval_helmholtz_split.py --weights path/to/weights.pt
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

BASE_DIR = Path(__file__).resolve().parent.parent

from ddpm.neural_networks.ddpm import GaussianDDPM
from ddpm.neural_networks.unets.unet_helmholtz_split import MyUNet_Helmholtz_Split
from ddpm.neural_networks.unets.unet_helmholtz_split_film import MyUNet_Helmholtz_Split_FiLM
from ddpm.helper_functions.standardize_data import ZScoreStandardizer, UnifiedZScoreStandardizer
from ddpm.utils.noise_utils import HelmholtzMatchedNoise
from ddpm.utils.helmholtz_split import helmholtz_decompose

# ── Constants ────────────────────────────────────────────────────────
OCEAN_H, OCEAN_W = 44, 94
FULL_H, FULL_W = 64, 128
N_STEPS = 250

U_MEAN, U_STD = -0.06929559429949586, 0.1358005549716049
V_MEAN, V_STD = -0.0323937796117541, 0.08899177232117582
SHARED_MEAN, SHARED_STD = -0.05084468695562498, 0.11479844598042026
NM = np.array([U_MEAN, V_MEAN], dtype=np.float32)
NS = np.array([U_STD, V_STD], dtype=np.float32)

DEFAULT_WEIGHTS = (
    "experiments/12_helmholtz_dual_head/helmholtz_split_decoder/results/"
    "inpaint_gaussian_t250_best_weights.pt"
)

# ── Args ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--coverage", type=float, nargs="+", default=[0.5],
                    help="Percent(s) of ocean cells observed (0.5 = 0.5%%)")
parser.add_argument("--n-samples", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS)
parser.add_argument("--t-vals", type=int, nargs="+",
                    default=[25, 50, 75, 100, 125],
                    help="Noise timesteps to sweep for single-step")
parser.add_argument("--rev-t-start", type=int, default=75,
                    help="Reverse chain start timestep")
parser.add_argument("--resample-steps", type=int, default=3,
                    help="RePaint resampling steps per timestep")
parser.add_argument("--guidance-scale", type=float, default=1.0,
                    help="DPS guidance scale for gradient-guided chain")
parser.add_argument("--skip-vcnn", action="store_true",
                    help="Skip V-CNN comparison")
parser.add_argument("--unified-std", action="store_true",
                    help="Use unified standardizer (for helmholtz_matched noise)")
args = parser.parse_args()

# ── Auto-detect standardizer and noise type from resolved config ─────
use_matched_noise = False
if not args.unified_std:
    cfg_path = Path(args.weights).parent / "resolved_config.yaml"
    if cfg_path.exists():
        import yaml
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        if cfg.get("noise_function") in ("helmholtz_matched", "div_free",
            "spectral_div_free", "forward_diff_div_free", "fwd_diff_eq_divfree"):
            args.unified_std = True
            print(f"Auto-detected unified standardizer from {cfg_path.name}")
        if cfg.get("noise_function") == "helmholtz_matched":
            use_matched_noise = True
            print("Auto-detected helmholtz_matched noise → using matched inference")

if args.unified_std:
    standardizer = UnifiedZScoreStandardizer(SHARED_MEAN, SHARED_STD)
else:
    standardizer = ZScoreStandardizer(U_MEAN, U_STD, V_MEAN, V_STD)

# ── Noise generator (matched or Gaussian) ────────────────────────────
_matched_gen = HelmholtzMatchedNoise() if use_matched_noise else None

def noise_fn(shape, device):
    """Generate noise matching the training distribution."""
    if _matched_gen is not None:
        return _matched_gen.generate(shape, device=device)
    return torch.randn(shape, device=device)

# ── Device ───────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Device: {device}")


# ── Data loading ─────────────────────────────────────────────────────
def load_val_data():
    import pickle
    with open(str(BASE_DIR / "data.pickle"), "rb") as f:
        _, _, test_raw = pickle.load(f)
    test = np.transpose(test_raw, (3, 2, 1, 0)).astype(np.float32)
    test = np.nan_to_num(test, nan=0.0)
    return torch.from_numpy(test)


def get_ocean_mask(val_tensor):
    return (val_tensor[0].abs().sum(dim=0) > 1e-7).float().numpy()


def make_border_mask(shape, dev):
    m = torch.zeros(1, 1, shape[2], shape[3], device=dev)
    m[:, :, :OCEAN_H, :OCEAN_W] = 1.0
    return m


def random_obs_mask(ocean_mask, pct, rng):
    idx = np.argwhere(ocean_mask > 0.5)
    n = max(1, round(len(idx) * pct / 100.0))
    sel = rng.choice(len(idx), size=n, replace=False)
    m = np.zeros((OCEAN_H, OCEAN_W), dtype=np.float32)
    for i in sel:
        m[idx[i][0], idx[i][1]] = 1.0
    return m


def voronoi_fill(vel_obs, obs_mask, ocean_mask):
    ky, kx = np.where(obs_mask > 0.5)
    if len(ky) == 0:
        return np.zeros_like(vel_obs)
    tree = cKDTree(np.stack([ky, kx], axis=1).astype(np.float64))
    gy, gx = np.mgrid[0:OCEAN_H, 0:OCEAN_W]
    _, idx = tree.query(np.stack([gy.ravel(), gx.ravel()], axis=1).astype(np.float64))
    idx = idx.reshape(OCEAN_H, OCEAN_W)
    filled = np.stack([vel_obs[0, ky, kx][idx], vel_obs[1, ky, kx][idx]], axis=0)
    return filled * ocean_mask


# ── V-CNN (optional) ────────────────────────────────────────────────
def load_vcnn(dev):
    from scripts.voronoi_cnn_model import VoronoiCNN
    ck = torch.load(str(BASE_DIR / "results/voronoi_cnn/voronoi_cnn_best.pt"),
                    map_location="cpu", weights_only=False)
    m = VoronoiCNN(**ck["model_config"]).to(dev)
    m.load_state_dict(ck["model_state"])
    m.eval()
    return m


def predict_vcnn(model, vel_obs, obs_mask, ocean_mask, dev):
    from scripts.voronoi_cnn_model import build_voronoi_input
    vel_n = ((vel_obs - NM[:, None, None]) / NS[:, None, None]) * ocean_mask[None]
    vi = build_voronoi_input(vel_n, obs_mask, ocean_mask)
    with torch.no_grad():
        p = model(torch.from_numpy(vi).unsqueeze(0).to(dev))
    mt = torch.tensor(NM).view(1, 2, 1, 1).to(dev)
    st = torch.tensor(NS).view(1, 2, 1, 1).to(dev)
    om = torch.from_numpy(ocean_mask).float().to(dev).view(1, 1, OCEAN_H, OCEAN_W)
    phys = (p * st + mt) * om
    return phys.squeeze(0).cpu().numpy()


# ── Model loading ────────────────────────────────────────────────────
is_film_model = False  # set during load_model()

def load_model(weights_path):
    global is_film_model
    # Auto-detect UNet type from resolved config
    cfg_path = Path(weights_path).parent / "resolved_config.yaml"
    unet_type = "helmholtz_split"  # default
    if cfg_path.exists():
        import yaml
        with open(cfg_path) as f:
            c = yaml.safe_load(f)
        unet_type = c.get("unet_type", "helmholtz_split")

    is_film_model = (unet_type == "helmholtz_split_film")

    if is_film_model:
        net = MyUNet_Helmholtz_Split_FiLM(n_steps=N_STEPS, time_emb_dim=256)
        print("Loaded FiLM-conditioned Helmholtz split UNet")
    else:
        net = MyUNet_Helmholtz_Split(n_steps=N_STEPS, time_emb_dim=256,
                                     n_stage_tokens=0, self_cond_channels=0)
        print("Loaded unconditional Helmholtz split UNet")

    ddpm = GaussianDDPM(net, n_steps=N_STEPS,
                        min_beta=0.0001, max_beta=0.02, device=device)
    state = torch.load(str(BASE_DIR / weights_path), map_location="cpu",
                       weights_only=False)
    ddpm.load_state_dict(state)
    ddpm.to(device)
    ddpm.eval()
    return ddpm


def _model_input(x_t, miss_mask, known_mask, known_std, vor_std=None):
    """Build network input: 2ch for unconditional, 5ch for FiLM.

    For FiLM models, applies mask_xt (replaces known region with noise)
    and concatenates [x_t_masked, miss_mask(1ch), cond(2ch)].
    Uses Voronoi fill (dense) as conditioning to avoid sparse-signal washout.

    Training convention:
      - miss_mask: 1=missing, 0=known  (matches dataset mask_single)
      - vor_std: Voronoi-interpolated field (dense), or known_std as fallback
    """
    if not is_film_model:
        return x_t
    # mask_xt: replace known region of x_t with independent noise
    noise_replace = torch.randn_like(x_t)
    x_t_masked = x_t * miss_mask + noise_replace * known_mask
    # miss channel: 1 where missing (matches training mask convention)
    miss_ch = miss_mask[:, :1]  # (1, 1, H, W) — both channels are identical
    # conditioning: use Voronoi fill (dense signal) for FiLM
    cond_field = vor_std if vor_std is not None else known_std * known_mask
    return torch.cat([x_t_masked, miss_ch, cond_field], dim=1)  # (1, 5, H, W)


# ── Mask building ────────────────────────────────────────────────────
def build_masks(gt_ocean, obs_mask, ocean_mask):
    """Return known_std, miss_mask, known_mask, vor_std (all on device)."""
    gt_full = np.zeros((1, 2, FULL_H, FULL_W), dtype=np.float32)
    gt_full[0, :, :OCEAN_H, :OCEAN_W] = gt_ocean * ocean_mask[None]
    known_std = standardizer(
        torch.from_numpy(gt_full).squeeze(0)).unsqueeze(0).to(device)

    border = make_border_mask((1, 2, FULL_H, FULL_W), device)
    land_mask = (torch.from_numpy(gt_full).abs() > 1e-5).float().to(device)
    raw_miss = np.ones((FULL_H, FULL_W), dtype=np.float32)
    raw_miss[:OCEAN_H, :OCEAN_W] -= obs_mask
    raw_miss[:OCEAN_H, :OCEAN_W] *= ocean_mask
    raw_miss[OCEAN_H:, :] = 0.0
    raw_miss[:, OCEAN_W:] = 0.0
    miss_mask = (torch.from_numpy(raw_miss).unsqueeze(0).unsqueeze(0).to(device)
                 * border * land_mask)
    known_mask = 1.0 - miss_mask

    vel_obs = gt_ocean * obs_mask[None]
    vor_fill = voronoi_fill(vel_obs, obs_mask, ocean_mask)
    vor_full = np.zeros((1, 2, FULL_H, FULL_W), dtype=np.float32)
    vor_full[0, :, :OCEAN_H, :OCEAN_W] = vor_fill
    vor_std = standardizer(
        torch.from_numpy(vor_full).squeeze(0)).unsqueeze(0).to(device)

    return known_std, miss_mask, known_mask, vor_std


# ── Single-step inference ────────────────────────────────────────────
def run_single_step(ddpm, gt_ocean, obs_mask, ocean_mask, seed, t_val):
    known_std, miss_mask, known_mask, vor_std = build_masks(
        gt_ocean, obs_mask, ocean_mask)
    current = known_std * known_mask + vor_std * miss_mask

    torch.manual_seed(seed)
    with torch.no_grad():
        time_tensor = torch.full((1, 1), t_val, device=device, dtype=torch.long)
        alpha_bar_t = ddpm.alpha_bars[t_val]
        eps = noise_fn(current.shape, device)
        x_t = alpha_bar_t.sqrt() * current + (1 - alpha_bar_t).sqrt() * eps
        x0_pred = ddpm.network(_model_input(x_t, miss_mask, known_mask, known_std, vor_std), time_tensor)
        result = known_std * known_mask + x0_pred * miss_mask

    result_phys = standardizer.unstandardize(result.squeeze(0).cpu())
    return result_phys[:, :OCEAN_H, :OCEAN_W].numpy()


# ── Reverse chain inference (corrected RePaint) ─────────────────────
def run_reverse_chain(ddpm, gt_ocean, obs_mask, ocean_mask, seed, t_start,
                      resample_steps=3):
    known_std, miss_mask, known_mask, vor_std = build_masks(
        gt_ocean, obs_mask, ocean_mask)
    current = known_std * known_mask + vor_std * miss_mask

    torch.manual_seed(seed)
    with torch.no_grad():
        alpha_bar_start = ddpm.alpha_bars[t_start]
        eps = noise_fn(current.shape, device)
        x = alpha_bar_start.sqrt() * current + (1 - alpha_bar_start).sqrt() * eps

        for t in range(t_start, -1, -1):
            n_resample = resample_steps if t > 0 else 1
            for r in range(n_resample):
                time_tensor = torch.full((1, 1), t, device=device, dtype=torch.long)
                x0_pred = ddpm.network(_model_input(x, miss_mask, known_mask, known_std, vor_std), time_tensor)

                if t > 0:
                    alpha_bar_t = ddpm.alpha_bars[t]
                    alpha_bar_prev = ddpm.alpha_bars[t - 1]
                    alpha_t = ddpm.alphas[t]
                    beta_t = ddpm.betas[t]
                    coef1 = (alpha_bar_prev.sqrt() * beta_t) / (1 - alpha_bar_t)
                    coef2 = (alpha_t.sqrt() * (1 - alpha_bar_prev)) / (1 - alpha_bar_t)
                    mean = coef1 * x0_pred + coef2 * x
                    var = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
                    noise = noise_fn(x.shape, device)
                    x_denoised = mean + var.sqrt() * noise

                    # Forward-noise known region to level t-1 before paste
                    noise_known = noise_fn(known_std.shape, device)
                    x_known_t = (alpha_bar_prev.sqrt() * known_std
                                 + (1 - alpha_bar_prev).sqrt() * noise_known)
                    x = x_known_t * known_mask + x_denoised * miss_mask
                else:
                    x_denoised = x0_pred
                    x = known_std * known_mask + x_denoised * miss_mask

                if r < n_resample - 1 and t > 0:
                    noise_back = noise_fn(x.shape, device)
                    x = alpha_t.sqrt() * x + (1 - alpha_t).sqrt() * noise_back

    result_phys = standardizer.unstandardize(x.squeeze(0).cpu())
    return result_phys[:, :OCEAN_H, :OCEAN_W].numpy()


# ── Gradient-guided reverse chain (DPS-style, x0-prediction) ────────
def run_guided_chain(ddpm, gt_ocean, obs_mask, ocean_mask, seed, t_start,
                     guidance_scale=1.0):
    """Reverse chain with gradient guidance + proper known-region noising.

    At each step:
      1. Predict x̂₀ = network(x_t, t)
      2. Compute boundary loss: L = ||( x̂₀ - known ) * known_mask||²
      3. Backprop to x_t → ∇_{x_t} L
      4. Shift posterior mean: μ ← μ − ζ · ∇_{x_t} L / ||residual||
      5. Sample x_{t-1} ~ N(μ, σ²I)
      6. Paste forward-noised known region (hard constraint)
    """
    known_std, miss_mask, known_mask, vor_std = build_masks(
        gt_ocean, obs_mask, ocean_mask)
    current = known_std * known_mask + vor_std * miss_mask

    for p in ddpm.parameters():
        p.requires_grad_(False)

    torch.manual_seed(seed)
    with torch.no_grad():
        alpha_bar_start = ddpm.alpha_bars[t_start]
        eps = noise_fn(current.shape, device)
        x = alpha_bar_start.sqrt() * current + (1 - alpha_bar_start).sqrt() * eps

    for t in range(t_start, -1, -1):
        x_in = x.detach().requires_grad_(True)
        time_tensor = torch.full((1, 1), t, device=device, dtype=torch.long)
        x0_pred = ddpm.network(_model_input(x_in, miss_mask, known_mask, known_std, vor_std), time_tensor)

        # Boundary loss on known region
        diff = (x0_pred - known_std) * known_mask
        loss = (diff ** 2).sum()
        residual_norm = diff.detach().norm().clamp(min=1e-8)
        grad = torch.autograd.grad(loss, x_in)[0]

        with torch.no_grad():
            x0_hat = x0_pred.detach()

            if t > 0:
                alpha_bar_t = ddpm.alpha_bars[t]
                alpha_bar_prev = ddpm.alpha_bars[t - 1]
                alpha_t = ddpm.alphas[t]
                beta_t = ddpm.betas[t]

                coef1 = (alpha_bar_prev.sqrt() * beta_t) / (1 - alpha_bar_t)
                coef2 = (alpha_t.sqrt() * (1 - alpha_bar_prev)) / (1 - alpha_bar_t)
                mean = coef1 * x0_hat + coef2 * x_in.detach()

                # DPS guidance: shift mean (only in missing region)
                mean = mean - guidance_scale * grad / residual_norm

                var = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
                z = noise_fn(x.shape, device)
                x_denoised = mean + var.sqrt() * z

                # Paste forward-noised known region at level t-1
                noise_known = noise_fn(known_std.shape, device)
                x_known_t = (alpha_bar_prev.sqrt() * known_std
                             + (1 - alpha_bar_prev).sqrt() * noise_known)
                x = x_known_t * known_mask + x_denoised * miss_mask
            else:
                x = known_std * known_mask + x0_hat * miss_mask

    result_phys = standardizer.unstandardize(x.squeeze(0).cpu())
    return result_phys[:, :OCEAN_H, :OCEAN_W].numpy()


# ── Helmholtz-projected reverse chain ────────────────────────────────
def run_helmholtz_chain(ddpm, gt_ocean, obs_mask, ocean_mask, seed, t_start,
                        resample_steps=3, sol_weight=1.0, irr_weight=0.0):
    """Reverse chain exploiting the Helmholtz decomposition.

    At each step, after x₀ prediction, extracts the solenoidal (div-free)
    component from the ψ head and optionally blends the irrotational
    component from the φ head with controllable weights. This constrains
    the reverse process to stay near the div-free manifold.
    """
    known_std, miss_mask, known_mask, vor_std = build_masks(
        gt_ocean, obs_mask, ocean_mask)
    current = known_std * known_mask + vor_std * miss_mask
    net = ddpm.network

    torch.manual_seed(seed)
    with torch.no_grad():
        alpha_bar_start = ddpm.alpha_bars[t_start]
        eps = noise_fn(current.shape, device)
        x = alpha_bar_start.sqrt() * current + (1 - alpha_bar_start).sqrt() * eps

        for t in range(t_start, -1, -1):
            n_resample = resample_steps if t > 0 else 1
            for r in range(n_resample):
                time_tensor = torch.full((1, 1), t, device=device, dtype=torch.long)
                _ = net(_model_input(x, miss_mask, known_mask, known_std, vor_std), time_tensor)  # populates last_v_sol, last_v_irr

                # Helmholtz-projected x₀: weighted blend of heads
                x0_proj = sol_weight * net.last_v_sol + irr_weight * net.last_v_irr

                if t > 0:
                    alpha_bar_t = ddpm.alpha_bars[t]
                    alpha_bar_prev = ddpm.alpha_bars[t - 1]
                    alpha_t = ddpm.alphas[t]
                    beta_t = ddpm.betas[t]
                    coef1 = (alpha_bar_prev.sqrt() * beta_t) / (1 - alpha_bar_t)
                    coef2 = (alpha_t.sqrt() * (1 - alpha_bar_prev)) / (1 - alpha_bar_t)
                    mean = coef1 * x0_proj + coef2 * x
                    var = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
                    noise = noise_fn(x.shape, device)
                    x_denoised = mean + var.sqrt() * noise

                    # Forward-noise known region to level t-1
                    noise_known = noise_fn(known_std.shape, device)
                    x_known_t = (alpha_bar_prev.sqrt() * known_std
                                 + (1 - alpha_bar_prev).sqrt() * noise_known)
                    x = x_known_t * known_mask + x_denoised * miss_mask
                else:
                    x = known_std * known_mask + x0_proj * miss_mask

                if r < n_resample - 1 and t > 0:
                    noise_back = noise_fn(x.shape, device)
                    x = alpha_t.sqrt() * x + (1 - alpha_t).sqrt() * noise_back

    result_phys = standardizer.unstandardize(x.squeeze(0).cpu())
    return result_phys[:, :OCEAN_H, :OCEAN_W].numpy()


# ── Ensemble single-step averaging ──────────────────────────────────
def run_ensemble(ddpm, gt_ocean, obs_mask, ocean_mask, seed, t_val,
                 n_ensemble=10):
    """Average N single-step predictions with different noise seeds."""
    known_std, miss_mask, known_mask, vor_std = build_masks(
        gt_ocean, obs_mask, ocean_mask)
    current = known_std * known_mask + vor_std * miss_mask

    x0_sum = torch.zeros_like(current)
    with torch.no_grad():
        for k in range(n_ensemble):
            torch.manual_seed(seed + k * 1000)
            time_tensor = torch.full((1, 1), t_val, device=device,
                                     dtype=torch.long)
            alpha_bar_t = ddpm.alpha_bars[t_val]
            eps = noise_fn(current.shape, device)
            x_t = alpha_bar_t.sqrt() * current + (1 - alpha_bar_t).sqrt() * eps
            x0_pred = ddpm.network(_model_input(x_t, miss_mask, known_mask, known_std, vor_std), time_tensor)
            x0_sum += x0_pred

    x0_avg = x0_sum / n_ensemble
    result = known_std * known_mask + x0_avg * miss_mask
    result_phys = standardizer.unstandardize(result.squeeze(0).cpu())
    return result_phys[:, :OCEAN_H, :OCEAN_W].numpy()


# ── Single-step div-free: use only the ψ head ───────────────────────
def run_single_divfree(ddpm, gt_ocean, obs_mask, ocean_mask, seed, t_val):
    """Single-step inference using only the solenoidal (div-free) head."""
    known_std, miss_mask, known_mask, vor_std = build_masks(
        gt_ocean, obs_mask, ocean_mask)
    current = known_std * known_mask + vor_std * miss_mask
    net = ddpm.network

    torch.manual_seed(seed)
    with torch.no_grad():
        time_tensor = torch.full((1, 1), t_val, device=device, dtype=torch.long)
        alpha_bar_t = ddpm.alpha_bars[t_val]
        eps = noise_fn(current.shape, device)
        x_t = alpha_bar_t.sqrt() * current + (1 - alpha_bar_t).sqrt() * eps
        _ = net(_model_input(x_t, miss_mask, known_mask, known_std, vor_std), time_tensor)  # populates last_v_sol

        # Use only the solenoidal component (guaranteed div-free)
        x0_sol = net.last_v_sol
        result = known_std * known_mask + x0_sol * miss_mask

    result_phys = standardizer.unstandardize(result.squeeze(0).cpu())
    return result_phys[:, :OCEAN_H, :OCEAN_W].numpy()


# ── Metrics ──────────────────────────────────────────────────────────
def ocean_mse(pred, gt, ocean_mask):
    ocean_b = ocean_mask.astype(bool)
    diff = pred[:, ocean_b] - gt[:, ocean_b]
    return float((diff ** 2).mean())


def finite_div(u, v):
    """Discrete divergence via central differences (m/s per grid cell)."""
    du_dx = np.zeros_like(u)
    dv_dy = np.zeros_like(v)
    du_dx[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / 2.0
    dv_dy[1:-1, :] = (v[2:, :] - v[:-2, :]) / 2.0
    return du_dx + dv_dy


def ocean_div_rms(pred, ocean_mask):
    d = finite_div(pred[0], pred[1])
    ocean_b = ocean_mask.astype(bool)
    return float(np.sqrt((d[ocean_b] ** 2).mean()))


# ══════════════════════════════════════════════════════════════════════
def main():
    coverages = args.coverage
    n_samples = args.n_samples
    t_vals = args.t_vals

    print(f"\n{'='*72}")
    print(f"HELMHOLTZ SPLIT-DECODER — EVALUATION")
    print(f"Weights: {args.weights}")
    print(f"Coverage: {coverages}%  |  Samples: {n_samples}  |  "
          f"t_vals: {t_vals}")
    print(f"Rev chain: t_start={args.rev_t_start}, resample={args.resample_steps}")
    print(f"{'='*72}")

    val = load_val_data()
    ocean_mask = get_ocean_mask(val)
    n_ocean = int(ocean_mask.sum())

    rng = np.random.default_rng(seed=7777)
    val_indices = rng.choice(val.shape[0], size=min(n_samples, val.shape[0]),
                             replace=False)
    val_indices.sort()

    # Load models
    print("\nLoading split-decoder model...")
    ddpm = load_model(args.weights)
    net = ddpm.network
    n_params = sum(p.numel() for p in net.parameters())
    print(f"  Split UNet: {n_params/1e6:.2f}M params")

    vcnn = None
    if not args.skip_vcnn:
        try:
            vcnn = load_vcnn(device)
            print("  V-CNN loaded")
        except Exception as e:
            print(f"  V-CNN not available ({e}), skipping")

    # Method list
    method_names = ["Voronoi"]
    if vcnn is not None:
        method_names.append("V-CNN")
    for tv in t_vals:
        method_names.append(f"1S@t={tv}")
    method_names.append("RevChain")
    method_names.append("Guided")
    method_names.append("HelmProj")
    method_names.append("Ens10")
    method_names.append("1S-DivFree")

    all_results = {}  # {coverage: {method: [mse_list]}}
    all_divs = {}     # {coverage: {method: [div_rms_list]}}
    all_preds = {}    # {coverage: {method: [pred_arrays]}}

    for cov in coverages:
        n_obs = max(1, round(n_ocean * cov / 100.0))
        print(f"\n{'─'*72}")
        print(f"Coverage: {cov}%  ({n_obs} / {n_ocean} ocean cells)")
        print(f"{'─'*72}")

        results = {m: [] for m in method_names}
        divs = {m: [] for m in method_names}
        preds = {m: [] for m in method_names}

        print(f"\n{'#':>3} {'Idx':>5}", end="")
        for m in method_names:
            print(f" {m:>13}", end="")
        print(f" {'DivRMS':>10} {'Time':>7}")
        print("-" * (10 + 14 * len(method_names) + 18))

        t0_global = time.time()
        for i, vi in enumerate(val_indices):
            gt = val[vi, :, :OCEAN_H, :OCEAN_W].numpy()
            seed = args.seed + vi
            obs_mask = random_obs_mask(ocean_mask, cov,
                                       np.random.default_rng(seed=seed))
            t0 = time.time()

            # Voronoi baseline
            vel_obs = gt * obs_mask[None]
            vor = voronoi_fill(vel_obs, obs_mask, ocean_mask)
            results["Voronoi"].append(ocean_mse(vor, gt, ocean_mask))
            divs["Voronoi"].append(ocean_div_rms(vor, ocean_mask))
            preds["Voronoi"].append(vor)

            # V-CNN
            if vcnn is not None:
                vcnn_pred = predict_vcnn(vcnn, vel_obs, obs_mask, ocean_mask, device)
                results["V-CNN"].append(ocean_mse(vcnn_pred, gt, ocean_mask))
                divs["V-CNN"].append(ocean_div_rms(vcnn_pred, ocean_mask))
                preds["V-CNN"].append(vcnn_pred)

            # Single-step at each t_val
            for tv in t_vals:
                ss = run_single_step(ddpm, gt, obs_mask, ocean_mask, seed, tv)
                results[f"1S@t={tv}"].append(ocean_mse(ss, gt, ocean_mask))
                divs[f"1S@t={tv}"].append(ocean_div_rms(ss, ocean_mask))
                preds[f"1S@t={tv}"].append(ss)

            # Reverse chain
            rc = run_reverse_chain(ddpm, gt, obs_mask, ocean_mask, seed,
                                   t_start=args.rev_t_start,
                                   resample_steps=args.resample_steps)
            results["RevChain"].append(ocean_mse(rc, gt, ocean_mask))
            divs["RevChain"].append(ocean_div_rms(rc, ocean_mask))
            preds["RevChain"].append(rc)

            # Gradient-guided chain
            gc = run_guided_chain(ddpm, gt, obs_mask, ocean_mask, seed,
                                  t_start=args.rev_t_start,
                                  guidance_scale=args.guidance_scale)
            results["Guided"].append(ocean_mse(gc, gt, ocean_mask))
            divs["Guided"].append(ocean_div_rms(gc, ocean_mask))
            preds["Guided"].append(gc)

            # Helmholtz-projected chain (solenoidal only)
            hc = run_helmholtz_chain(ddpm, gt, obs_mask, ocean_mask, seed,
                                     t_start=args.rev_t_start,
                                     resample_steps=args.resample_steps,
                                     sol_weight=1.0, irr_weight=0.0)
            results["HelmProj"].append(ocean_mse(hc, gt, ocean_mask))
            divs["HelmProj"].append(ocean_div_rms(hc, ocean_mask))
            preds["HelmProj"].append(hc)

            # Ensemble single-step (average 10 predictions)
            best_t = t_vals[0] if len(t_vals) == 1 else 50
            es = run_ensemble(ddpm, gt, obs_mask, ocean_mask, seed,
                              t_val=best_t, n_ensemble=10)
            results["Ens10"].append(ocean_mse(es, gt, ocean_mask))
            divs["Ens10"].append(ocean_div_rms(es, ocean_mask))
            preds["Ens10"].append(es)

            # Single-step div-free (solenoidal head only)
            sf = run_single_divfree(ddpm, gt, obs_mask, ocean_mask, seed,
                                    t_val=best_t)
            results["1S-DivFree"].append(ocean_mse(sf, gt, ocean_mask))
            divs["1S-DivFree"].append(ocean_div_rms(sf, ocean_mask))
            preds["1S-DivFree"].append(sf)

            elapsed = time.time() - t0
            row = f"{i+1:>3} {vi:>5}"
            for m in method_names:
                row += f" {results[m][-1]:>13.6f}"
            row += f" {divs['RevChain'][-1]:>10.6f}"
            row += f" {elapsed:>6.1f}s"
            print(row)

        total = time.time() - t0_global
        all_results[cov] = results
        all_divs[cov] = divs
        all_preds[cov] = preds

        # Summary for this coverage
        print(f"\n  SUMMARY — {cov}% coverage, {len(val_indices)} samples, "
              f"{total:.0f}s total")
        ref_key = "V-CNN" if vcnn is not None else "Voronoi"
        ref_mean = np.mean(results[ref_key])
        print(f"  {'Method':<20} {'Mean MSE':<14} {'vs '+ref_key:<12} "
              f"{'Div RMS':<12}")
        print(f"  {'-'*56}")
        for m in method_names:
            arr = np.array(results[m])
            darr = np.array(divs[m])
            ratio = arr.mean() / ref_mean if ref_mean > 0 else float("inf")
            print(f"  {m:<20} {arr.mean():.7f}     {ratio:.3f}x"
                  f"       {darr.mean():.6f}")

    # ── Helmholtz decomposition diagnostics ──────────────────────────
    print(f"\n{'='*72}")
    print("HELMHOLTZ DECOMPOSITION DIAGNOSTICS")
    print(f"{'='*72}")
    if hasattr(net, 'last_psi') and net.last_psi is not None:
        psi = net.last_psi
        phi = net.last_phi
        v_sol = net.last_v_sol
        v_irr = net.last_v_irr

        psi_rms = float(psi.detach().pow(2).mean().sqrt())
        phi_rms = float(phi.detach().pow(2).mean().sqrt())
        sol_rms = float(v_sol.detach().pow(2).mean().sqrt())
        irr_rms = float(v_irr.detach().pow(2).mean().sqrt())
        total_rms = float((v_sol + v_irr).detach().pow(2).mean().sqrt())
        ratio = phi_rms / (psi_rms + 1e-12)

        print(f"  ψ RMS:     {psi_rms:.6f}")
        print(f"  φ RMS:     {phi_rms:.6f}")
        print(f"  φ/ψ ratio: {ratio:.4f}")
        print(f"  v_sol RMS: {sol_rms:.6f}")
        print(f"  v_irr RMS: {irr_rms:.6f}")
        print(f"  total RMS: {total_rms:.6f}")

        # Check for cancellation degeneracy
        cancel_ratio = (sol_rms + irr_rms) / (total_rms + 1e-12)
        if cancel_ratio > 3.0:
            print(f"  ⚠ CANCELLATION WARNING: sol+irr / total = {cancel_ratio:.1f}x")
        else:
            print(f"  ✓ Healthy decomposition: sol+irr / total = {cancel_ratio:.2f}x")

        # Cosine similarity between heads
        vs = v_sol.detach().reshape(-1)
        vi = v_irr.detach().reshape(-1)
        cos = float(torch.dot(vs, vi) / (vs.norm() * vi.norm() + 1e-12))
        print(f"  cos(v_sol, v_irr): {cos:.4f}  "
              f"({'anti-correlated' if cos < -0.3 else 'near-orthogonal' if abs(cos) < 0.3 else 'correlated'})")
    else:
        print("  (no Helmholtz diagnostics — run forward first)")

    # ── Per-head evaluation vs ground truth decomposition ────────────
    print(f"\n{'='*72}")
    print("PER-HEAD EVALUATION vs GROUND TRUTH HELMHOLTZ DECOMPOSITION")
    print(f"{'='*72}")
    print("  Running single-step (best t) on test samples, comparing each")
    print("  network head to the FFT Helmholtz decomposition of ground truth...\n")

    # Use the best t_val from evaluation
    cov0 = coverages[0]
    best_t = min(t_vals, key=lambda t: np.mean(all_results[cov0][f"1S@t={t}"]))
    print(f"  Using t={best_t} (best single-step timestep at {cov0}% coverage)")

    head_stats = {"sol_mse": [], "irr_mse": [], "total_mse": [],
                  "sol_cos": [], "irr_cos": [],
                  "gt_sol_frac": [], "pred_sol_frac": [],
                  "cancel_ratio": []}

    n_head_samples = min(len(val_indices), 10)
    print(f"\n  {'#':>3} {'Idx':>5}  {'Head→GT_sol':>11} {'Head→GT_irr':>11} "
          f"{'Total MSE':>11}  {'cos_sol':>8} {'cos_irr':>8}  "
          f"{'GT sol%':>7} {'Pred sol%':>9} {'Cancel':>7}")
    print(f"  {'-'*100}")

    for i in range(n_head_samples):
        vi = val_indices[i]
        gt = val[vi, :, :OCEAN_H, :OCEAN_W].numpy()
        seed = args.seed + vi
        obs_mask = random_obs_mask(ocean_mask, cov0,
                                   np.random.default_rng(seed=seed))

        # Run single step to populate net.last_v_sol / last_v_irr
        known_std, miss_mask, known_mask, vor_std = build_masks(
            gt, obs_mask, ocean_mask)
        current = known_std * known_mask + vor_std * miss_mask

        torch.manual_seed(seed)
        with torch.no_grad():
            time_tensor = torch.full((1, 1), best_t, device=device,
                                     dtype=torch.long)
            alpha_bar_t = ddpm.alpha_bars[best_t]
            eps = noise_fn(current.shape, device)
            x_t = alpha_bar_t.sqrt() * current + (1 - alpha_bar_t).sqrt() * eps
            x0_pred = ddpm.network(_model_input(x_t, miss_mask, known_mask, known_std, vor_std), time_tensor)

        # Get network's head outputs (in standardized space, full grid)
        pred_v_sol = net.last_v_sol.detach().cpu()  # (1, 2, H, W)
        pred_v_irr = net.last_v_irr.detach().cpu()

        # Ground truth in standardized space (full grid)
        gt_full = np.zeros((1, 2, FULL_H, FULL_W), dtype=np.float32)
        gt_full[0, :, :OCEAN_H, :OCEAN_W] = gt * ocean_mask[None]
        gt_std = standardizer(
            torch.from_numpy(gt_full).squeeze(0)).unsqueeze(0)

        # FFT Helmholtz decomposition of ground truth
        gt_sol, gt_irr = helmholtz_decompose(gt_std)

        # MSE of each head vs its GT counterpart (ocean region only)
        om = torch.zeros(1, 1, FULL_H, FULL_W)
        om[:, :, :OCEAN_H, :OCEAN_W] = torch.from_numpy(ocean_mask).float()
        n_ocean_px = om.sum() * 2  # 2 channels

        sol_mse = float(((pred_v_sol - gt_sol).pow(2) * om).sum() / n_ocean_px)
        irr_mse = float(((pred_v_irr - gt_irr).pow(2) * om).sum() / n_ocean_px)
        total_mse = float(((x0_pred.cpu() - gt_std).pow(2) * om).sum() / n_ocean_px)

        # Cosine similarity of each head vs GT (ocean region, flattened)
        def _cos(a, b, mask):
            a_flat = (a * mask).reshape(-1)
            b_flat = (b * mask).reshape(-1)
            return float(torch.dot(a_flat, b_flat) /
                         (a_flat.norm() * b_flat.norm() + 1e-12))

        sol_cos = _cos(pred_v_sol, gt_sol, om)
        irr_cos = _cos(pred_v_irr, gt_irr, om)

        # Energy fractions
        gt_sol_energy = float((gt_sol.pow(2) * om).sum())
        gt_irr_energy = float((gt_irr.pow(2) * om).sum())
        gt_total_energy = gt_sol_energy + gt_irr_energy + 1e-12
        gt_sol_frac = gt_sol_energy / gt_total_energy

        pred_sol_energy = float((pred_v_sol.pow(2) * om).sum())
        pred_irr_energy = float((pred_v_irr.pow(2) * om).sum())
        pred_total_energy = pred_sol_energy + pred_irr_energy + 1e-12
        pred_sol_frac = pred_sol_energy / pred_total_energy

        # Cancellation
        pred_sol_rms = float((pred_v_sol.pow(2) * om).sum().sqrt())
        pred_irr_rms = float((pred_v_irr.pow(2) * om).sum().sqrt())
        pred_total_rms = float(((pred_v_sol + pred_v_irr).pow(2) * om).sum().sqrt())
        cancel = (pred_sol_rms + pred_irr_rms) / (pred_total_rms + 1e-12)

        head_stats["sol_mse"].append(sol_mse)
        head_stats["irr_mse"].append(irr_mse)
        head_stats["total_mse"].append(total_mse)
        head_stats["sol_cos"].append(sol_cos)
        head_stats["irr_cos"].append(irr_cos)
        head_stats["gt_sol_frac"].append(gt_sol_frac)
        head_stats["pred_sol_frac"].append(pred_sol_frac)
        head_stats["cancel_ratio"].append(cancel)

        print(f"  {i+1:>3} {vi:>5}  {sol_mse:>11.6f} {irr_mse:>11.6f} "
              f"{total_mse:>11.6f}  {sol_cos:>8.4f} {irr_cos:>8.4f}  "
              f"{gt_sol_frac*100:>6.1f}% {pred_sol_frac*100:>8.1f}% "
              f"{cancel:>6.1f}x")

    print(f"\n  AVERAGES:")
    print(f"    Head→GT_sol MSE:  {np.mean(head_stats['sol_mse']):.6f}")
    print(f"    Head→GT_irr MSE:  {np.mean(head_stats['irr_mse']):.6f}")
    print(f"    Total MSE:        {np.mean(head_stats['total_mse']):.6f}")
    print(f"    cos(ψ-head, GT_sol): {np.mean(head_stats['sol_cos']):.4f}")
    print(f"    cos(φ-head, GT_irr): {np.mean(head_stats['irr_cos']):.4f}")
    print(f"    GT sol fraction:  {np.mean(head_stats['gt_sol_frac'])*100:.1f}%")
    print(f"    Pred sol fraction: {np.mean(head_stats['pred_sol_frac'])*100:.1f}%")
    print(f"    Cancel ratio:     {np.mean(head_stats['cancel_ratio']):.1f}x")

    # ── Save results ─────────────────────────────────────────────────
    out_dir = Path(args.weights).parent
    save_dict = {
        "coverages": coverages,
        "val_indices": val_indices,
        "method_names": method_names,
        "results": all_results,
        "divs": all_divs,
        "args": vars(args),
    }
    save_path = out_dir / "eval_split_decoder.pt"
    torch.save(save_dict, str(BASE_DIR / save_path))
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
