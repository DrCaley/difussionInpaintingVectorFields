#!/usr/bin/env python3
"""Topo-aware model evaluation via GP-Diff S6 pipeline.

Loads existing per-sample data from exp07 bulk eval (GT, obs_mask, GP output)
and runs the topology-aware unconditional DDPM through the same S6 GP-init
adaptive RePaint pipeline. Compares against existing GP, V-CNN, and baseline
GP-Diff results.

Usage:
    PYTHONPATH=. python tmp_topo_gpdiff_eval.py
    PYTHONPATH=. python tmp_topo_gpdiff_eval.py --coverage 1.0
    PYTHONPATH=. python tmp_topo_gpdiff_eval.py --n-samples 10
"""

import argparse, os, sys, time
from pathlib import Path
import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from ddpm.neural_networks.ddpm import GaussianDDPM
from ddpm.neural_networks.unets.unet_xl_attn import MyUNet_Attn
from ddpm.helper_functions.standardize_data import ZScoreStandardizer
from ddpm.helper_functions.masks.border_mask import BorderMaskGenerator
from ddpm.utils.inpainting_utils import repaint_gp_init_adaptive
from ddpm.utils.noise_utils import get_noise_strategy

# ── constants ────────────────────────────────────────────────────────
OCEAN_H, OCEAN_W = 44, 94
FULL_H, FULL_W   = 64, 128
N_STEPS = 250

U_MEAN, U_STD = -0.06929559429949586, 0.1358005549716049
V_MEAN, V_STD = -0.0323937796117541, 0.08899177232117582
standardizer = ZScoreStandardizer(U_MEAN, U_STD, V_MEAN, V_STD)

# S6 default parameters (matching exp07 bulk_eval/run.py)
S6_DEFAULTS = dict(
    max_stages=6,
    t_start=75,
    t_refine=50,
    resample_steps=5,
    noise_floor=0.2,
    noise_floor_refine=0.3,
    var_decay=0.1,
    gamma=3.0,
    seed=42,
)

# Model paths
TOPO_WEIGHTS = (
    "experiments/10_topology_metrics/topo_aware_training/results/"
    "inpaint_gaussian_t250_best_ema_weights.pt"
)
BASELINE_CKPT = (
    "experiments/02_inpaint_algorithm/repaint_gaussian_attn/results/"
    "inpaint_gaussian_t250_best_checkpoint.pt"
)

# Existing eval results directory
EVAL_BASE = "experiments/07_ddpm_composite/bulk_eval/results"

COVERAGE_TAGS = {
    5.0: "5.0pct_n100_tc75_s6",
    1.0: "1.0pct_n100_tc200_s6",
    0.1: "0.1pct_n100_tc200_s6",
}


# ── args ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--coverage", type=float, default=5.0,
                    help="Coverage %% to evaluate (5.0, 1.0, or 0.1)")
parser.add_argument("--n-samples", type=int, default=0,
                    help="Number of samples (0 = all available)")
parser.add_argument("--also-baseline", action="store_true",
                    help="Also re-run baseline model for verification")
args = parser.parse_args()

# ── device ───────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Device: {device}")


# ── load model ───────────────────────────────────────────────────────
def load_uncond_model(weight_path, is_checkpoint=False):
    """Load unconditional MyUNet_Attn DDPM."""
    if is_checkpoint:
        ckpt = torch.load(str(BASE_DIR / weight_path), map_location="cpu",
                          weights_only=False)
        n_steps  = ckpt.get("n_steps", 250)
        min_beta = ckpt.get("min_beta", 0.0001)
        max_beta = ckpt.get("max_beta", 0.02)
        net = MyUNet_Attn(n_steps=n_steps, time_emb_dim=256)
        ddpm = GaussianDDPM(net, n_steps=n_steps,
                            min_beta=min_beta, max_beta=max_beta, device=device)
        ddpm.load_state_dict(ckpt["model_state_dict"])
    else:
        net = MyUNet_Attn(n_steps=N_STEPS, time_emb_dim=256)
        ddpm = GaussianDDPM(net, n_steps=N_STEPS,
                            min_beta=0.0001, max_beta=0.02, device=device)
        state = torch.load(str(BASE_DIR / weight_path), map_location="cpu",
                           weights_only=False)
        ddpm.load_state_dict(state)
    ddpm.to(device)
    ddpm.eval()
    return ddpm


# ── S6 RePaint ───────────────────────────────────────────────────────
noise_strategy = get_noise_strategy("gaussian")
border_gen = BorderMaskGenerator()


def run_s6(ddpm_model, gt_np, obs_mask, ocean_mask, gp_mean, gp_var,
           sample_seed):
    """Run S6 multi-stage GP-Diff on a single sample. Returns (2,44,94) numpy."""

    # Build full-size GT in physical space
    gt_full = np.zeros((1, 2, FULL_H, FULL_W), dtype=np.float32)
    gt_full[0, :, :OCEAN_H, :OCEAN_W] = gt_np * ocean_mask[None]
    input_image = standardizer(
        torch.from_numpy(gt_full).squeeze(0)).unsqueeze(0).to(device)

    # Build full-size GP in standardized space
    gp_full = np.zeros((1, 2, FULL_H, FULL_W), dtype=np.float32)
    gp_full[0, :, :OCEAN_H, :OCEAN_W] = gp_mean * ocean_mask[None]
    gp_std = standardizer(
        torch.from_numpy(gp_full).squeeze(0)).unsqueeze(0).to(device)

    # Missing mask (1=missing)
    land_mask = (torch.from_numpy(gt_full).abs() > 1e-5).float().to(device)
    raw_miss = np.ones((FULL_H, FULL_W), dtype=np.float32)
    raw_miss[:OCEAN_H, :OCEAN_W] -= obs_mask
    raw_miss[:OCEAN_H, :OCEAN_W] *= ocean_mask
    raw_miss[OCEAN_H:, :] = 0.0
    raw_miss[:, OCEAN_W:] = 0.0
    raw_miss_t = torch.from_numpy(raw_miss).unsqueeze(0).unsqueeze(0).to(device)
    border = border_gen.generate_mask(
        torch.Size([1, 2, FULL_H, FULL_W])).to(device)
    missing_mask = raw_miss_t * border * land_mask

    # GP variance (physical space)
    gp_var_full = np.zeros((1, 2, FULL_H, FULL_W), dtype=np.float32)
    gp_var_full[0, :, :OCEAN_H, :OCEAN_W] = gp_var * ocean_mask[None]
    gp_var_t = torch.from_numpy(gp_var_full).to(device)

    # Multi-stage refinement
    p = S6_DEFAULTS
    current_prior = gp_std.clone()
    current_var = gp_var_t.clone()

    for stage in range(1, p["max_stages"] + 1):
        if stage == 1:
            t_s, nf = p["t_start"], p["noise_floor"]
            seed_s = sample_seed
        else:
            t_s, nf = p["t_refine"], p["noise_floor_refine"]
            seed_s = sample_seed + stage * 10000
            current_var = current_var * p["var_decay"]

        torch.manual_seed(seed_s)
        with torch.no_grad():
            stage_out = repaint_gp_init_adaptive(
                ddpm_model, input_image, missing_mask,
                gp_image=current_prior,
                gp_variance_map=current_var,
                t_start=t_s,
                noise_floor=nf,
                n_samples=1, device=device,
                noise_strategy=noise_strategy,
                prediction_target="eps",
                resample_steps=p["resample_steps"],
                project_div_free=False,
                anneal_floor=False,
                gamma=p["gamma"],
            )
        current_prior = stage_out.clone()

    # Convert to physical space, crop to ocean
    result_phys = standardizer.unstandardize(stage_out.squeeze(0).cpu())
    return result_phys[:, :OCEAN_H, :OCEAN_W].numpy()


def ocean_mse(pred, gt, ocean_mask):
    """MSE over ocean cells. pred, gt: (2, H, W) numpy."""
    ocean_b = ocean_mask.astype(bool)
    diff = pred[:, ocean_b] - gt[:, ocean_b]
    return float((diff ** 2).mean())


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    tag = COVERAGE_TAGS.get(args.coverage)
    if tag is None:
        print(f"No existing results for coverage={args.coverage}%")
        print(f"Available: {list(COVERAGE_TAGS.keys())}")
        return

    eval_dir = BASE_DIR / EVAL_BASE / tag
    summary = torch.load(str(eval_dir / "summary.pt"),
                         map_location="cpu", weights_only=False)
    records = summary["records"]
    n_total = len(records)
    n_eval = args.n_samples if args.n_samples > 0 else n_total

    # Randomly select which samples to evaluate (deterministic seed for
    # reproducibility).  When n_eval == n_total we still shuffle so the
    # first-10 quick test isn't biased toward low-index samples.
    rng = np.random.default_rng(seed=12345)
    eval_indices = rng.choice(n_total, size=min(n_eval, n_total),
                              replace=False)
    eval_indices.sort()  # process in order for easier directory lookup
    n_eval = len(eval_indices)

    print(f"\n{'='*80}")
    print(f"TOPO-AWARE GP-DIFF (S6) EVALUATION")
    print(f"Coverage: {args.coverage}%  |  Samples: {n_eval}/{n_total}")
    if n_eval < n_total:
        print(f"Randomly selected indices: {eval_indices.tolist()}")
    print(f"{'='*80}")

    # Load topo model
    print("\nLoading topology-aware model (EMA)...")
    topo_ddpm = load_uncond_model(TOPO_WEIGHTS, is_checkpoint=False)

    baseline_ddpm = None
    if args.also_baseline:
        print("Loading baseline model (exp02)...")
        baseline_ddpm = load_uncond_model(BASELINE_CKPT, is_checkpoint=True)

    # Collect results
    results = {
        "gp_mse": [], "vcnn_mse": [], "gpdiff_mse": [],
        "topo_gpdiff_mse": [],
    }
    if args.also_baseline:
        results["base_gpdiff_mse"] = []

    print(f"\n{'#':>4} {'SmpIdx':>7} {'ValIdx':>7} {'GP MSE':>12} {'VCNN MSE':>12} "
          f"{'GPDiff MSE':>12} {'Topo MSE':>12} {'Topo/VCNN':>10} {'Time':>7}")
    print("-" * 95)

    t0_global = time.time()

    for count, i in enumerate(eval_indices):
        rec = records[i]
        vi = rec["val_idx"]
        sample_seed = S6_DEFAULTS["seed"] + i

        # Load per-sample data
        sample_dirs = sorted(eval_dir.glob(f"sample_{i:03d}_vi{vi}"))
        if not sample_dirs:
            print(f"  [SKIP] sample_{i:03d}_vi{vi} not found")
            continue
        tensors = torch.load(str(sample_dirs[0] / "tensors.pt"),
                             map_location="cpu", weights_only=False)

        gt = np.asarray(tensors["gt"])          # (2, 44, 94)
        obs_mask = np.asarray(tensors["obs_mask"])  # (44, 94)
        ocean_mask = np.asarray(tensors["ocean_mask"])  # (44, 94)
        gp_mean = np.asarray(tensors["gp_mean"])  # (2, 44, 94)
        gp_var = np.asarray(tensors["gp_var"])    # (2, 44, 94)

        # Existing MSEs
        mse = tensors["mse"]
        gp_mse = mse["gp_mse"]
        vcnn_mse = mse["vcnn_mse"]
        gpdiff_mse = mse["gpdiff_mse"]

        # Run topo model S6
        t0 = time.time()
        topo_pred = run_s6(topo_ddpm, gt, obs_mask, ocean_mask,
                           gp_mean, gp_var, sample_seed)
        topo_mse = ocean_mse(topo_pred, gt, ocean_mask)
        elapsed = time.time() - t0

        ratio_vcnn = topo_mse / vcnn_mse if vcnn_mse > 0 else float("inf")

        print(f"{count+1:>4} {i:>7} {vi:>7} {gp_mse:>12.8f} {vcnn_mse:>12.8f} "
              f"{gpdiff_mse:>12.8f} {topo_mse:>12.8f} {ratio_vcnn:>9.3f}x "
              f"{elapsed:>6.1f}s")

        results["gp_mse"].append(gp_mse)
        results["vcnn_mse"].append(vcnn_mse)
        results["gpdiff_mse"].append(gpdiff_mse)
        results["topo_gpdiff_mse"].append(topo_mse)

        if baseline_ddpm:
            base_pred = run_s6(baseline_ddpm, gt, obs_mask, ocean_mask,
                               gp_mean, gp_var, sample_seed)
            base_mse = ocean_mse(base_pred, gt, ocean_mask)
            results["base_gpdiff_mse"].append(base_mse)

    elapsed_total = time.time() - t0_global

    # ── Summary ──────────────────────────────────────────────────────
    n = len(results["topo_gpdiff_mse"])
    gp = np.array(results["gp_mse"])
    vcnn = np.array(results["vcnn_mse"])
    gpdiff = np.array(results["gpdiff_mse"])
    topo = np.array(results["topo_gpdiff_mse"])

    # Save results
    out_path = (BASE_DIR / "experiments/10_topology_metrics/topo_aware_training/"
                f"results/topo_gpdiff_eval_{args.coverage}pct.pt")
    torch.save({
        "coverage_pct": args.coverage,
        "n_samples": n,
        "results": results,
        "s6_params": S6_DEFAULTS,
        "topo_weights": TOPO_WEIGHTS,
    }, str(out_path))
    print(f"\nResults saved to {out_path}")

    print(f"\n{'='*80}")
    print(f"SUMMARY: {args.coverage}% coverage, {n} samples, "
          f"{elapsed_total:.0f}s total")
    print(f"{'='*80}")

    methods = {"GP": gp, "V-CNN": vcnn, "GP-Diff (base)": gpdiff,
               "GP-Diff (topo)": topo}
    if args.also_baseline:
        base = np.array(results["base_gpdiff_mse"])
        methods["GP-Diff (re-run)"] = base

    print(f"\n{'Method':<20} {'Mean MSE':>12} {'Median MSE':>12} "
          f"{'Std':>12} {'Wins':>6}")
    print("-" * 65)

    all_methods = list(methods.items())
    for name, arr in all_methods:
        wins = sum(1 for j in range(n) if all(
            arr[j] <= methods[other_name][j]
            for other_name, other_arr in all_methods if other_name != name))
        print(f"{name:<20} {arr.mean():>12.8f} {np.median(arr):>12.8f} "
              f"{arr.std():>12.8f} {wins:>6}")

    # Head-to-head: Topo vs V-CNN
    topo_beats_vcnn = sum(1 for j in range(n) if topo[j] < vcnn[j])
    topo_beats_gpdiff = sum(1 for j in range(n) if topo[j] < gpdiff[j])
    vcnn_beats_gpdiff = sum(1 for j in range(n) if vcnn[j] < gpdiff[j])

    print(f"\nHead-to-head ({n} samples):")
    print(f"  Topo beats V-CNN:    {topo_beats_vcnn}/{n} "
          f"({100*topo_beats_vcnn/n:.1f}%)")
    print(f"  Topo beats GP-Diff:  {topo_beats_gpdiff}/{n} "
          f"({100*topo_beats_gpdiff/n:.1f}%)")
    print(f"  V-CNN beats GP-Diff: {vcnn_beats_gpdiff}/{n} "
          f"({100*vcnn_beats_gpdiff/n:.1f}%)")

    print(f"\nMean MSE ratios:")
    print(f"  Topo / GP:     {topo.mean() / gp.mean():.4f}x")
    print(f"  Topo / V-CNN:  {topo.mean() / vcnn.mean():.4f}x")
    print(f"  Topo / GP-Diff:{topo.mean() / gpdiff.mean():.4f}x")
    print(f"  V-CNN / GP:    {vcnn.mean() / gp.mean():.4f}x")
    print(f"  GP-Diff / GP:  {gpdiff.mean() / gp.mean():.4f}x")


if __name__ == "__main__":
    main()
