#!/usr/bin/env python3

from __future__ import annotations

import random
import sys
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from scipy import stats


BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR.parent))

from data_prep.data_initializer import DDInitializer
from ddpm.helper_functions.interpolation_tool import gp_fill
from ddpm.helper_functions.masks.straigth_line import StraightLineMaskGenerator
from ddpm.helper_functions.compute_divergence import compute_divergence
from ddpm.neural_networks.ddpm import GaussianDDPM
from ddpm.neural_networks.unets.unet_xl import MyUNet
from ddpm.utils.inpainting_utils import inpaint_generate_new_images
from ddpm.utils.noise_utils import get_noise_strategy


RANDOM_SEED = 26
NUM_RANDOM_SAMPLES = 50
MASK_NUM_LINES = 1
MASK_LINE_THICKNESS = 3
MODEL_PATHS = [
	BASE_DIR.parent / "trained_models" / "weekend_ddpm_ocean_model.pt",
	BASE_DIR.parent / "trained_models" / "div_free_model.pt",
]


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)


def masked_mse(predicted: torch.Tensor, target: torch.Tensor, missing_mask: torch.Tensor) -> float:
	target = target.to(predicted.device)
	missing_mask = missing_mask.to(predicted.device)
	per_pixel = ((predicted - target) ** 2).sum(dim=1, keepdim=True)
	total = (per_pixel * missing_mask).sum()
	denom = missing_mask.sum()
	if denom.item() == 0:
		return float("nan")
	return float((total / denom).item())


def load_diffusion_model(dd: DDInitializer, model_path: Path) -> tuple[GaussianDDPM, object]:
	checkpoint = torch.load(model_path, map_location=dd.get_device(), weights_only=False)
	model_state_dict = checkpoint.get("model_state_dict", checkpoint)
	n_steps = checkpoint.get("n_steps", dd.get_attribute("noise_steps"))
	min_beta = checkpoint.get("min_beta", dd.get_attribute("min_beta"))
	max_beta = checkpoint.get("max_beta", dd.get_attribute("max_beta"))
	standardizer_strategy = checkpoint.get("standardizer_strategy", dd.get_standardizer())
	noise_strategy = checkpoint.get("noise_strategy")
	if noise_strategy is None:
		noise_name = "div_free" if "div_free" in model_path.stem else "gaussian"
		noise_strategy = get_noise_strategy(noise_name)

	dd.reinitialize(min_beta, max_beta, n_steps, standardizer_strategy)

	model = GaussianDDPM(MyUNet(n_steps), n_steps=n_steps, min_beta=min_beta, max_beta=max_beta, device=dd.get_device())
	model.load_state_dict(model_state_dict)
	model.eval()
	return model, noise_strategy


def build_gp_control_prediction(sample: torch.Tensor, missing_mask: torch.Tensor) -> torch.Tensor:
	observed = sample * (1.0 - missing_mask).repeat(1, 2, 1, 1)
	gp_filled = gp_fill(observed.clone().cpu(), missing_mask.repeat(1, 2, 1, 1).cpu())
	return gp_filled.to(sample.device)


def build_diffusion_prediction(
	model: GaussianDDPM,
	sample: torch.Tensor,
	missing_mask: torch.Tensor,
	noise_strategy,
	standardizer,
	diffusion_device: torch.device,
) -> torch.Tensor:
	input_image = sample.to(diffusion_device)
	mask = missing_mask.to(diffusion_device)
	output = inpaint_generate_new_images(
		model,
		input_image,
		mask,
		n_samples=1,
		device=diffusion_device,
		resample_steps=1,
		noise_strategy=noise_strategy,
		return_final_noised=False,
	)
	output = torch.unsqueeze(standardizer.unstandardize(torch.squeeze(output, 0)).to(diffusion_device), 0)
	return output


def evaluate_model(
	dd: DDInitializer,
	model_path: Path,
	sample_indices: list[int],
	mask_generator: StraightLineMaskGenerator,
	device: torch.device,
	masks: list | None = None,
) -> dict[str, list[float] | float | str]:
	model, noise_strategy = load_diffusion_model(dd, model_path)
	standardizer = dd.get_standardizer()
	test_data = dd.get_test_data()

	model_mse_scores: list[float] = []
	model_mag_rmse_scores: list[float] = []
	model_divergence_rmse_scores: list[float] = []

	for i, sample_index in enumerate(sample_indices):
		sample_standardized, _, _ = test_data[sample_index]
		sample_standardized = torch.nan_to_num(sample_standardized).float()
		sample = standardizer.unstandardize(sample_standardized).unsqueeze(0).cpu().float()
		sample = torch.nan_to_num(sample)

		# reuse pre-generated masks when provided to ensure both models see identical masks
		if masks is not None:
			known_mask = masks[i].cpu().float()
		else:
			known_mask = mask_generator.generate_mask(sample.shape).cpu().float()
		missing_mask = 1.0 - known_mask

		noise_for_model = noise_strategy if noise_strategy is not None else get_noise_strategy("gaussian")
		model_prediction = build_diffusion_prediction(
			model,
			sample_standardized.unsqueeze(0),
			missing_mask,
			noise_for_model,
			standardizer,
			device,
		)
		model_prediction = torch.nan_to_num(model_prediction)

		model_mse = masked_mse(model_prediction, sample, missing_mask)
		model_mse_scores.append(model_mse)

		# magnitude RMSE on missing region
		pred_u = model_prediction[0, 0]
		pred_v = model_prediction[0, 1]
		gt_u = sample[0, 0].to(pred_u.device)
		gt_v = sample[0, 1].to(pred_v.device)
		mask2 = missing_mask.squeeze(0).squeeze(0).to(pred_u.device)
		mag_pred = torch.sqrt(pred_u ** 2 + pred_v ** 2)
		mag_gt = torch.sqrt(gt_u ** 2 + gt_v ** 2)
		mag_rmse = torch.sqrt(((mag_pred - mag_gt) ** 2 * mask2).sum() / mask2.sum()) if mask2.sum().item() > 0 else float('nan')
		model_mag_rmse_scores.append(float(mag_rmse.item()))

		# divergence RMSE on missing region
		div_pred = compute_divergence(pred_u, pred_v)
		div_gt = compute_divergence(gt_u, gt_v)
		div_rmse = torch.sqrt(((div_pred - div_gt) ** 2 * mask2).sum() / mask2.sum()) if mask2.sum().item() > 0 else float('nan')
		model_divergence_rmse_scores.append(float(div_rmse.item()))

		print(
			f"[{model_path.name}] sample={sample_index} model_mse={model_mse:.6f} "
			f"noise_strategy={type(noise_for_model).__name__}"
		)

	return {
		"model_path": str(model_path),
		"noise_strategy_name": type(noise_for_model).__name__,
		"model_mse_scores": model_mse_scores,
		"mean_model_mse": float(mean(model_mse_scores)) if model_mse_scores else float("nan"),
		"model_mag_rmse_scores": model_mag_rmse_scores,
		"mean_model_mag_rmse": float(mean(model_mag_rmse_scores)) if model_mag_rmse_scores else float("nan"),
		"model_divergence_rmse_scores": model_divergence_rmse_scores,
		"mean_model_divergence_rmse": float(mean(model_divergence_rmse_scores)) if model_divergence_rmse_scores else float("nan"),
	}


def main() -> dict[str, dict[str, list[float] | float | str]]:
	dd = DDInitializer()
	set_seed(RANDOM_SEED)
	device = dd.get_device()

	test_data = dd.get_test_data()
	total_samples = len(test_data)
	sample_indices = random.sample(range(total_samples), k=min(NUM_RANDOM_SAMPLES, total_samples))

	# Initialize mask generator and pre-generate masks for selected samples
	mask_generator = StraightLineMaskGenerator(num_lines=MASK_NUM_LINES, line_thickness=MASK_LINE_THICKNESS)
	masks = []
	for idx in sample_indices:
		sample_standardized, _, _ = test_data[idx]
		h, w = sample_standardized.shape[1], sample_standardized.shape[2]
		m = mask_generator.generate_mask((1, 1, h, w))
		masks.append(m)

	results: dict[str, dict[str, list[float] | float | str]] = {}
	for model_path in MODEL_PATHS:
		results[model_path.name] = evaluate_model(dd, model_path, sample_indices, mask_generator, device, masks=masks)

	if len(MODEL_PATHS) == 2:
		first_name, second_name = MODEL_PATHS[0].name, MODEL_PATHS[1].name
		first_scores = results[first_name]["model_mse_scores"]
		second_scores = results[second_name]["model_mse_scores"]
		paired_differences = [second - first for first, second in zip(first_scores, second_scores)]
		results["paired_comparison"] = {
			"paired_differences": paired_differences,
			"mean_paired_difference": float(mean(paired_differences)) if paired_differences else float("nan"),
			"paired_t_test": stats.ttest_rel(second_scores, first_scores, nan_policy="omit") if paired_differences else None,
			"wilcoxon_test": stats.wilcoxon(second_scores, first_scores, zero_method="wilcox") if paired_differences else None,
		}
		print(f"\nPaired comparison ({second_name} - {first_name})")
		print(paired_differences)
		print(f"Mean paired difference: {results['paired_comparison']['mean_paired_difference']:.6f}")
		paired_t = results["paired_comparison"]["paired_t_test"]
		wilcoxon_t = results["paired_comparison"]["wilcoxon_test"]
		if paired_t is not None:
			print(f"Paired t-test: statistic={paired_t.statistic:.6f}, pvalue={paired_t.pvalue:.6g}")
		if wilcoxon_t is not None:
			print(f"Wilcoxon signed-rank: statistic={wilcoxon_t.statistic:.6f}, pvalue={wilcoxon_t.pvalue:.6g}")

	return results


if __name__ == "__main__":
	main()
