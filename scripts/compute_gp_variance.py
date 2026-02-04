import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from data_prep.data_initializer import DDInitializer


def main():
    dd = DDInitializer()
    device = dd.get_device()

    num_images = dd.get_attribute("gp_autocorr_num_images") or 10
    loader = DataLoader(
        dd.get_training_data(),
        batch_size=1,
        shuffle=False,
        num_workers=dd.get_attribute("num_workers") or 0,
    )

    u_vals = []
    v_vals = []

    image_counter = 0
    for batch in loader:
        if image_counter >= num_images:
            break

        input_image = batch[0].to(device)
        input_image_original = dd.get_standardizer().unstandardize(torch.squeeze(input_image, 0)).to(device)
        input_image_original = torch.unsqueeze(input_image_original, 0)

        valid_mask = (input_image_original.abs().sum(dim=1, keepdim=True) > 1e-5).float()
        valid_mask = valid_mask[0, 0].bool()

        u = input_image_original[0, 0][valid_mask].detach().cpu().numpy()
        v = input_image_original[0, 1][valid_mask].detach().cpu().numpy()

        if u.size > 0:
            u_vals.append(u)
        if v.size > 0:
            v_vals.append(v)

        image_counter += 1

    if not u_vals or not v_vals:
        raise RuntimeError("No valid data found for variance estimation.")

    u_all = np.concatenate(u_vals)
    v_all = np.concatenate(v_vals)

    u_var = float(np.var(u_all))
    v_var = float(np.var(v_all))
    combined_var = float(np.var(np.concatenate([u_all, v_all])))

    results = {
        "num_images": image_counter,
        "u_var": u_var,
        "v_var": v_var,
        "combined_var": combined_var,
    }

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "gp_variance_estimate.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
