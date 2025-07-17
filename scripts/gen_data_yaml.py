import os

import numpy as np
import oyaml as yaml

from pathlib import Path

pkg_path = Path(__file__).resolve().parent.parent

def main():
    with open(pkg_path / "data.yaml") as f:
        data_yaml = yaml.load(f, Loader=yaml.SafeLoader)

    run_dir = pkg_path / 'models_to_train'
    os.makedirs(run_dir, exist_ok=True)

    print(run_dir)

    for divergence_weight in np.arange(0, 1, 0.2):
        mse_weight = 1 - divergence_weight

        data_yaml['w1'] = float(divergence_weight)
        data_yaml['w2'] = float(mse_weight)

        with open(run_dir / f"{data_yaml['noise_function']}mse_{mse_weight:.1f}_div_{divergence_weight:.1f}.yaml", 'w') as f:
            yaml.dump(data_yaml, f)

if __name__ == '__main__':
    main()