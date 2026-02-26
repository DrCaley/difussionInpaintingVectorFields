#!/usr/bin/env python
"""
DDPM vs GP inpainting on TRAINING data (multiple samples).

Uses the same inference pipeline as test_attn_inpainting.py (repaint_standard)
with RobotPathGenerator masks, but draws from the training set instead of
validation — to test whether the "overfit" model performs well on memorized data.

Usage:
    PYTHONPATH=. python scripts/run_ddpm_gp_training_data.py
    PYTHONPATH=. python scripts/run_ddpm_gp_training_data.py --num-samples 10
    PYTHONPATH=. python scripts/run_ddpm_gp_training_data.py --checkpoint <path>
"""
import argparse
import random
import sys
from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from data_prep.data_initializer import DDInitializer
from ddpm.neural_networks.ddpm import GaussianDDPM
from ddpm.neural_networks.unets.unet_xl_attn import MyUNet_Attn
from ddpm.helper_functions.masks.robot_path import RobotPathGenerator
from ddpm.utils.inpainting_utils import repaint_standard
from ddpm.utils.noise_utils import get_noise_strategy
from ddpm.helper_functions.interpolation_tool import gp_fill

DEFAULT_CKPT = (
    "experiments/02_inpaint_algorithm/repaint_gaussian_attn/results/"
    "inpaint_gaussian_t250_Feb20_0722.pt"
)

RESAMPLE_STEPS = 5


def run_one_sample(idx, train_data, ddpm, standardizer, noise_strategy, dd, device, n_steps):
    """Run DDPM + GP inpainting on a single training sample. Returns dict of results."""
    input_image = train_data[idx][0].unsqueeze(0).to(device)
    input_orig = standardizer.unstandardize(
        input_image.squeeze(0)
    ).to(device).unsqueeze(0)

    land_mask = (input_orig.abs() > 1e-5).float().to(device)
    raw_mask  = RobotPathGenerator().generate_mask(input_image.shape).to(device)
    missing_mask = raw_mask * land_mask

    mask_pct = missing_mask[:, 0:1].sum() / (land_mask[:, 0:1].sum() + 1e-8) * 100

    # RePaint
    with torch.no_grad():
        repaint_out = repaint_standard(
            ddpm, input_image, missing_mask,
            n_samples=1, device=device,
            noise_strategy=noise_strategy,
            prediction_target="eps",
            resample_steps=RESAMPLE_STEPS,
            project_div_free=False,
            project_final_steps=0,
        )
    repaint_phys = standardizer.unstandardize(
        repaint_out.squeeze(0)
    ).to(device).unsqueeze(0)

    diff = (repaint_phys - input_orig) * missing_mask
    ddpm_mse = (diff ** 2).sum() / (missing_mask.sum() + 1e-8)

    # GP
    gp_out = gp_fill(
        input_orig, missing_mask,
        lengthscale=dd.get_attribute("gp_lengthscale"),
        variance=dd.get_attribute("gp_variance"),
        noise=dd.get_attribute("gp_noise"),
        use_double=True,
        kernel_type=dd.get_attribute("gp_kernel_type"),
        coord_system=dd.get_attribute("gp_coord_system"),
    )
    diff_gp = (gp_out - input_orig) * missing_mask
    gp_mse = (diff_gp ** 2).sum() / (missing_mask.sum() + 1e-8)

    return {
        "train_index":  idx,
        "mask_pct":     mask_pct.item(),
        "ddpm_mse":     ddpm_mse.item(),
        "gp_mse":       gp_mse.item(),
        "gt":           input_orig.cpu(),
        "missing_mask": missing_mask.cpu(),
        "ddpm_out":     repaint_phys.cpu(),
        "gp_out":       gp_out.cpu(),
    }


def main():
    parser = argparse.ArgumentParser(description="DDPM vs GP on training data")
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Number of training samples to test (default: 10)")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dd = DDInitializer()
    device = dd.get_device()
    standardizer = dd.get_standardizer()
    noise_strategy = get_noise_strategy("gaussian")
    print(f"Device: {device}")

    # ── load checkpoint ──────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    n_steps  = ckpt.get("n_steps", 250)
    min_beta = ckpt.get("min_beta", 0.0001)
    max_beta = ckpt.get("max_beta", 0.02)
    epoch    = ckpt.get("epoch", "?")

    network = MyUNet_Attn(n_steps=n_steps, time_emb_dim=256)
    ddpm = GaussianDDPM(
        network, n_steps=n_steps,
        min_beta=min_beta, max_beta=max_beta,
        device=device,
    )
    ddpm.load_state_dict(ckpt["model_state_dict"])
    ddpm = ddpm.to(device)
    ddpm.eval()
    print(f"Loaded checkpoint: epoch={epoch}, best_test_loss={ckpt.get('best_test_loss', '?')}")

    # ── select training indices ──────────────────────────────────
    train_data = dd.get_training_data()
    n_total = len(train_data)
    n_samples = min(args.num_samples, n_total)
    # spread indices evenly across the training set
    indices = [int(i * n_total / n_samples) for i in range(n_samples)]
    print(f"Training set: {n_total} samples, testing {n_samples} at indices {indices}")

    # ── run all samples ──────────────────────────────────────────
    all_results = []
    for i, idx in enumerate(indices):
        print(f"\n--- Sample {i+1}/{n_samples} (train index {idx}) ---")
        result = run_one_sample(idx, train_data, ddpm, standardizer,
                                noise_strategy, dd, device, n_steps)
        winner = "DDPM" if result["ddpm_mse"] < result["gp_mse"] else "GP"
        ratio = result["ddpm_mse"] / result["gp_mse"] if result["gp_mse"] > 0 else float("inf")
        print(f"  Mask: {result['mask_pct']:.1f}% | "
              f"DDPM: {result['ddpm_mse']:.6f} | "
              f"GP: {result['gp_mse']:.6f} | "
              f"Ratio: {ratio:.2f}x | {winner} wins")
        all_results.append(result)

    # ── summary ──────────────────────────────────────────────────
    ddpm_mses = [r["ddpm_mse"] for r in all_results]
    gp_mses   = [r["gp_mse"]   for r in all_results]
    ddpm_wins = sum(1 for r in all_results if r["ddpm_mse"] < r["gp_mse"])

    avg_ddpm = sum(ddpm_mses) / len(ddpm_mses)
    avg_gp   = sum(gp_mses) / len(gp_mses)
    avg_ratio = avg_ddpm / avg_gp if avg_gp > 0 else float("inf")

    print()
    print("=" * 70)
    print(f"TRAINING DATA INPAINTING SUMMARY  ({n_samples} samples, seed={args.seed})")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Epoch      : {epoch}")
    print("-" * 70)
    print(f"  {'Index':>6}  {'Mask%':>6}  {'DDPM MSE':>10}  {'GP MSE':>10}  {'Ratio':>7}  Winner")
    print("-" * 70)
    for r in all_results:
        ratio = r["ddpm_mse"] / r["gp_mse"] if r["gp_mse"] > 0 else float("inf")
        winner = "DDPM" if r["ddpm_mse"] < r["gp_mse"] else "GP"
        print(f"  {r['train_index']:>6}  {r['mask_pct']:>5.1f}%  "
              f"{r['ddpm_mse']:>10.6f}  {r['gp_mse']:>10.6f}  {ratio:>6.2f}x  {winner}")
    print("-" * 70)
    print(f"  {'AVG':>6}  {'':>6}  {avg_ddpm:>10.6f}  {avg_gp:>10.6f}  {avg_ratio:>6.2f}x  "
          f"DDPM wins {ddpm_wins}/{n_samples}")
    print("=" * 70)

    # ── save .pt ─────────────────────────────────────────────────
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    pt_path = out_dir / "ddpm_vs_gp_training_10samples.pt"
    torch.save({
        "results":        [{k: v for k, v in r.items()
                            if k not in ("gt", "missing_mask", "ddpm_out", "gp_out")}
                           for r in all_results],
        "ddpm_mses":      ddpm_mses,
        "gp_mses":        gp_mses,
        "avg_ddpm_mse":   avg_ddpm,
        "avg_gp_mse":     avg_gp,
        "ddpm_wins":      ddpm_wins,
        "n_samples":      n_samples,
        "epoch":          epoch,
        "n_steps":        n_steps,
        "resample_steps": RESAMPLE_STEPS,
        "seed":           args.seed,
        "data_split":     "TRAINING",
        "indices":        indices,
    }, pt_path)
    print(f"\nSaved summary to {pt_path}")


if __name__ == "__main__":
    main()
