import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from data_prep.data_initializer import DDInitializer


def _corr_for_lag(field: torch.Tensor, mask: torch.Tensor, lag: int, axis: int) -> float:
    if axis == 1:  # x direction (width)
        f1 = field[:, :-lag]
        f2 = field[:, lag:]
        m = mask[:, :-lag] * mask[:, lag:]
    else:  # y direction (height)
        f1 = field[:-lag, :]
        f2 = field[lag:, :]
        m = mask[:-lag, :] * mask[lag:, :]

    if m.sum() == 0:
        return float("nan")

    f1 = f1[m.bool()]
    f2 = f2[m.bool()]

    f1_mean = f1.mean()
    f2_mean = f2.mean()
    cov = ((f1 - f1_mean) * (f2 - f2_mean)).mean()
    std = f1.std() * f2.std() + 1e-12
    return (cov / std).item()


def _e_fold_length(lags: np.ndarray, corr: np.ndarray, target: float = math.exp(-1)) -> float:
    valid = np.isfinite(corr)
    lags = lags[valid]
    corr = corr[valid]
    if len(lags) < 2:
        return float("nan")

    for i in range(1, len(corr)):
        if corr[i] <= target <= corr[i - 1]:
            x0, x1 = lags[i - 1], lags[i]
            y0, y1 = corr[i - 1], corr[i]
            if y1 == y0:
                return float(x1)
            t = (target - y0) / (y1 - y0)
            return float(x0 + t * (x1 - x0))

    return float("nan")


def main():
    dd = DDInitializer()
    device = dd.get_device()

    num_images = dd.get_attribute("gp_autocorr_num_images") or dd.get_attribute("gp_tuning_num_images") or 10
    max_lag = dd.get_attribute("gp_autocorr_max_lag") or 40

    loader = DataLoader(
        dd.get_training_data(),
        batch_size=1,
        shuffle=False,
        num_workers=dd.get_attribute("num_workers") or 0,
    )

    corrs = {
        "u_x": [],
        "u_y": [],
        "v_x": [],
        "v_y": [],
    }

    image_counter = 0
    for batch in loader:
        if image_counter >= num_images:
            break

        input_image = batch[0].to(device)
        input_image_original = dd.get_standardizer().unstandardize(torch.squeeze(input_image, 0)).to(device)
        input_image_original = torch.unsqueeze(input_image_original, 0)

        valid_mask = (input_image_original.abs().sum(dim=1, keepdim=True) > 1e-5).float()
        valid_mask = valid_mask[0, 0].cpu()

        u = input_image_original[0, 0].cpu()
        v = input_image_original[0, 1].cpu()

        for lag in range(1, max_lag + 1):
            corrs["u_x"].append(_corr_for_lag(u, valid_mask, lag, axis=1))
            corrs["u_y"].append(_corr_for_lag(u, valid_mask, lag, axis=0))
            corrs["v_x"].append(_corr_for_lag(v, valid_mask, lag, axis=1))
            corrs["v_y"].append(_corr_for_lag(v, valid_mask, lag, axis=0))

        image_counter += 1

    lags = np.arange(1, max_lag + 1)
    results = {}

    for key in corrs:
        values = np.array(corrs[key], dtype=np.float64).reshape(image_counter, max_lag)
        mean_corr = np.nanmean(values, axis=0)
        length = _e_fold_length(lags, mean_corr)
        results[key] = {
            "mean_corr": mean_corr.tolist(),
            "e_fold_length": length,
        }

    results["summary"] = {
        "num_images": image_counter,
        "max_lag": max_lag,
        "u_length_x": results["u_x"]["e_fold_length"],
        "u_length_y": results["u_y"]["e_fold_length"],
        "v_length_x": results["v_x"]["e_fold_length"],
        "v_length_y": results["v_y"]["e_fold_length"],
    }

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "autocorr_lengthscales.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("Autocorrelation length scales (e-folding):")
    print(json.dumps(results["summary"], indent=2))
    print(f"Saved detailed results to {out_path}")


if __name__ == "__main__":
    main()
