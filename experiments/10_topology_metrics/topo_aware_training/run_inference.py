#!/usr/bin/env python3
"""
Inference for the topology-aware DDPM (Experiment 10).

Loads the topo_aware_uncond_eps model (standard_attn UNet, eps-prediction,
gaussian noise, T=250) and runs RePaint inpainting on validation samples.

Compares against GP baseline and the standard DDPM from experiment 08
(repaint_gaussian_attn) to see if topology-aware training helps.

Usage:
    PYTHONPATH=. python experiments/10_topology_metrics/topo_aware_training/run_inference.py
    PYTHONPATH=. python experiments/10_topology_metrics/topo_aware_training/run_inference.py --n-samples 20
    PYTHONPATH=. python experiments/10_topology_metrics/topo_aware_training/run_inference.py --use-ema
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from data_prep.data_initializer import DDInitializer
from ddpm.neural_networks.ddpm import GaussianDDPM
from ddpm.neural_networks.unets.unet_xl_attn import MyUNet_Attn
from ddpm.helper_functions.masks.robot_path import RobotPathGenerator
from ddpm.utils.inpainting_utils import repaint_standard
from ddpm.utils.noise_utils import get_noise_strategy
from ddpm.helper_functions.interpolation_tool import gp_fill

# ── Paths ──
TOPO_CKPT = BASE_DIR / "experiments/10_topology_metrics/topo_aware_training/results/inpaint_gaussian_t250_best_checkpoint.pt"
TOPO_EMA  = BASE_DIR / "experiments/10_topology_metrics/topo_aware_training/results/inpaint_gaussian_t250_best_ema_weights.pt"
BASELINE_CKPT = BASE_DIR / "experiments/08_network_architecture/repaint_gaussian_attn/inpaint_gaussian_t250_Feb20_2305.pt"
RESULTS_DIR = BASE_DIR / "experiments/10_topology_metrics/topo_aware_training/results"

RESAMPLE_STEPS = 5


def load_model(ckpt_path, device, use_ema=False, ema_ckpt_path=None):
    """Load a standard_attn model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    n_steps  = ckpt.get("n_steps", 250)
    min_beta = ckpt.get("min_beta", 0.0001)
    max_beta = ckpt.get("max_beta", 0.02)
    epoch    = ckpt.get("epoch", "?")
    best     = ckpt.get("best_test_loss", "?")

    network = MyUNet_Attn(n_steps=n_steps, time_emb_dim=256)
    ddpm = GaussianDDPM(
        network, n_steps=n_steps,
        min_beta=min_beta, max_beta=max_beta,
        device=device,
    )

    if use_ema and ema_ckpt_path is not None and Path(ema_ckpt_path).exists():
        # Load EMA weights directly
        ema_weights = torch.load(ema_ckpt_path, map_location="cpu", weights_only=False)
        ddpm.load_state_dict(ema_weights)
        print(f"  Loaded EMA weights from {Path(ema_ckpt_path).name}")
    elif use_ema and "ema_state" in ckpt:
        # Load EMA shadow from checkpoint
        ddpm.load_state_dict(ckpt["ema_state"]["shadow"])
        print(f"  Loaded EMA shadow from checkpoint")
    else:
        ddpm.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded model_state_dict")

    ddpm = ddpm.to(device)
    ddpm.eval()
    print(f"  epoch={epoch}, best_test_loss={best}, n_steps={n_steps}")
    return ddpm, n_steps


def run_sample(ddpm, idx, val_data, standardizer, noise_strategy, dd, device,
               prediction_target="eps"):
    """Run RePaint inpainting on one validation sample."""
    input_image = val_data[idx][0].unsqueeze(0).to(device)
    input_orig = standardizer.unstandardize(
        input_image.squeeze(0)
    ).to(device).unsqueeze(0)

    land_mask = (input_orig.abs() > 1e-5).float().to(device)
    raw_mask = RobotPathGenerator().generate_mask(input_image.shape).to(device)
    missing_mask = raw_mask * land_mask
    mask_pct = missing_mask[:, 0:1].sum() / (land_mask[:, 0:1].sum() + 1e-8) * 100

    with torch.no_grad():
        repaint_out = repaint_standard(
            ddpm, input_image, missing_mask,
            n_samples=1, device=device,
            noise_strategy=noise_strategy,
            prediction_target=prediction_target,
            resample_steps=RESAMPLE_STEPS,
            project_div_free=False,
            project_final_steps=0,
        )

    repaint_phys = standardizer.unstandardize(
        repaint_out.squeeze(0)
    ).to(device).unsqueeze(0)

    diff = (repaint_phys - input_orig) * missing_mask
    ddpm_mse = (diff ** 2).sum() / (missing_mask.sum() + 1e-8)

    # GP baseline
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
        "val_index":    idx,
        "mask_pct":     mask_pct.item(),
        "ddpm_mse":     ddpm_mse.item(),
        "gp_mse":       gp_mse.item(),
        "gt":           input_orig.cpu(),
        "missing_mask": missing_mask.cpu(),
        "ddpm_out":     repaint_phys.cpu(),
        "gp_out":       gp_out.cpu(),
    }


def main():
    parser = argparse.ArgumentParser(description="Topology-aware DDPM inference")
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--use-ema", action="store_true", default=True,
                        help="Use EMA weights (default: True)")
    parser.add_argument("--no-ema", action="store_true",
                        help="Don't use EMA weights")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip standard DDPM baseline comparison")
    args = parser.parse_args()

    if args.no_ema:
        args.use_ema = False

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dd = DDInitializer()
    device = dd.get_device()
    standardizer = dd.get_standardizer()
    noise_strategy = get_noise_strategy("gaussian")
    val_data = dd.get_validation_data()

    n_total = len(val_data)
    n_samples = min(args.n_samples, n_total)
    # Spread evenly across validation set
    indices = [int(i * n_total / n_samples) for i in range(n_samples)]

    print(f"\n{'='*80}")
    print(f"TOPOLOGY-AWARE DDPM INFERENCE (Experiment 10)")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Samples: {n_samples}, EMA: {args.use_ema}")
    print(f"Val indices: {indices}")

    # ── Load topology-aware model ──
    print(f"\n--- Loading topology-aware model ---")
    topo_ddpm, n_steps = load_model(
        TOPO_CKPT, device,
        use_ema=args.use_ema,
        ema_ckpt_path=TOPO_EMA,
    )

    # ── Load baseline model (exp08) ──
    baseline_ddpm = None
    if not args.skip_baseline and BASELINE_CKPT.exists():
        print(f"\n--- Loading baseline model (exp08 repaint_gaussian_attn) ---")
        baseline_ddpm, _ = load_model(BASELINE_CKPT, device, use_ema=False)
    elif not args.skip_baseline:
        print(f"\n[WARN] Baseline checkpoint not found: {BASELINE_CKPT}")
        print("  Skipping baseline DDPM comparison.")

    # ── Run inference ──
    print(f"\n{'='*80}")
    print(f"Running RePaint inpainting (resample_steps={RESAMPLE_STEPS})...")
    print(f"{'='*80}\n")

    topo_results = []
    baseline_results = []

    for i, idx in enumerate(indices):
        print(f"--- Sample {i+1}/{n_samples} (val index {idx}) ---")

        t0 = time.time()
        result = run_sample(topo_ddpm, idx, val_data, standardizer,
                            noise_strategy, dd, device, prediction_target="eps")
        dt = time.time() - t0
        topo_results.append(result)

        ratio = result["ddpm_mse"] / result["gp_mse"] if result["gp_mse"] > 0 else float("inf")
        winner = "TOPO" if result["ddpm_mse"] < result["gp_mse"] else "GP"
        print(f"  TOPO:  MSE={result['ddpm_mse']:.6f} | GP: {result['gp_mse']:.6f} | "
              f"ratio={ratio:.3f}x | mask={result['mask_pct']:.1f}% | {winner} wins | {dt:.1f}s")

        if baseline_ddpm is not None:
            t0 = time.time()
            b_result = run_sample(baseline_ddpm, idx, val_data, standardizer,
                                  noise_strategy, dd, device, prediction_target="eps")
            dt = time.time() - t0
            baseline_results.append(b_result)

            b_ratio = b_result["ddpm_mse"] / b_result["gp_mse"] if b_result["gp_mse"] > 0 else float("inf")
            b_winner = "BASE" if b_result["ddpm_mse"] < b_result["gp_mse"] else "GP"
            print(f"  BASE:  MSE={b_result['ddpm_mse']:.6f} | GP: {b_result['gp_mse']:.6f} | "
                  f"ratio={b_ratio:.3f}x | {b_winner} wins | {dt:.1f}s")

    # ── Summary ──
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}\n")

    topo_mses = [r["ddpm_mse"] for r in topo_results]
    gp_mses = [r["gp_mse"] for r in topo_results]
    mask_pcts = [r["mask_pct"] for r in topo_results]

    avg_topo = np.mean(topo_mses)
    avg_gp = np.mean(gp_mses)
    topo_wins = sum(1 for t, g in zip(topo_mses, gp_mses) if t < g)

    print(f"Mask coverage:    {np.mean(mask_pcts):.1f}% +/- {np.std(mask_pcts):.1f}%")
    print(f"GP baseline:      MSE = {avg_gp:.6f}")
    print(f"Topo-DDPM:        MSE = {avg_topo:.6f}  ({avg_topo/avg_gp:.3f}x GP)  "
          f"wins {topo_wins}/{n_samples}")

    if baseline_results:
        base_mses = [r["ddpm_mse"] for r in baseline_results]
        avg_base = np.mean(base_mses)
        base_wins = sum(1 for b, g in zip(base_mses, gp_mses) if b < g)
        topo_vs_base = sum(1 for t, b in zip(topo_mses, base_mses) if t < b)

        print(f"Baseline DDPM:    MSE = {avg_base:.6f}  ({avg_base/avg_gp:.3f}x GP)  "
              f"wins {base_wins}/{n_samples}")
        print(f"\nTopo vs Baseline: Topo wins {topo_vs_base}/{n_samples}")
        print(f"  Topo improvement: {(1 - avg_topo/avg_base)*100:.1f}%")

    # ── Save results ──
    save_path = RESULTS_DIR / "topo_inference_results.pt"
    save_data = {
        "topo_results": topo_results,
        "baseline_results": baseline_results,
        "config": {
            "n_samples": n_samples,
            "use_ema": args.use_ema,
            "seed": args.seed,
            "resample_steps": RESAMPLE_STEPS,
            "prediction_target": "eps",
        },
        "summary": {
            "avg_topo_mse": avg_topo,
            "avg_gp_mse": avg_gp,
            "topo_vs_gp_ratio": avg_topo / avg_gp,
            "topo_wins_vs_gp": topo_wins,
        },
    }
    if baseline_results:
        save_data["summary"]["avg_baseline_mse"] = avg_base
        save_data["summary"]["baseline_vs_gp_ratio"] = avg_base / avg_gp
        save_data["summary"]["topo_wins_vs_baseline"] = topo_vs_base
        save_data["summary"]["topo_improvement_pct"] = (1 - avg_topo/avg_base)*100

    torch.save(save_data, save_path)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
