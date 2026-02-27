#!/usr/bin/env python
"""Launch an experiment by merging a base template with experiment overrides.

Usage
-----
    PYTHONPATH=. python experiments/run_experiment.py experiments/01_noise_strategy/fwd_divfree/config.yaml

What it does
------------
1. Loads the base template (``experiments/templates/base_inpaint.yaml``).
2. Deep-merges the experiment's ``config.yaml`` on top.
3. Validates component compatibility using ``ddpm.protocols``.
4. Writes the resolved config to the experiment's ``results/`` folder.
5. Launches ``ddpm/training/train_inpaint.py`` with the resolved config.

You can also do a dry-run (validate + print config without training):

    PYTHONPATH=. python experiments/run_experiment.py --dry-run experiments/01_.../config.yaml
"""

import argparse
import copy
import os
import subprocess
import sys
import yaml
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # project root
TEMPLATE_DIR = BASE_DIR / "experiments" / "templates"
DEFAULT_TEMPLATE = TEMPLATE_DIR / "base_inpaint.yaml"


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (override wins)."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def validate_config(config: dict) -> list[str]:
    """Run protocol-level compatibility checks on a resolved config.

    Returns a list of issue strings (empty = all good).
    """
    from ddpm.utils.noise_utils import get_noise_strategy
    from ddpm.helper_functions.standardize_data import STANDARDIZER_REGISTRY
    from ddpm.protocols import validate_all
    import inspect

    noise_fn_name = config.get("noise_function", "gaussian")
    noise_strategy = get_noise_strategy(noise_fn_name)

    # Resolve standardizer type
    std_type = config.get("standardizer_type", "zscore")
    if std_type == "auto":
        mapping = config.get("standardizer_by_noise", {})
        std_type = mapping.get(noise_fn_name, "zscore")

    std_class = STANDARDIZER_REGISTRY.get(std_type)
    if std_class is None:
        return [f"Unknown standardizer_type: {std_type}"]

    # Instantiate standardizer with dummy params
    sig = inspect.signature(std_class.__init__)
    init_params = list(sig.parameters.keys())[1:]
    try:
        args = [config[p] for p in init_params]
        standardizer = std_class(*args)
    except KeyError as e:
        return [f"Missing config key for standardizer {std_type}: {e}"]

    prediction_target = config.get("prediction_target", "eps")

    issues = validate_all(
        noise_strategy=noise_strategy,
        standardizer=standardizer,
        prediction_target=prediction_target,
    )

    # Additional checks
    unet_type = config.get("unet_type", "concat")
    if unet_type == "standard" and config.get("mask_xt", False):
        issues.append(
            "unet_type='standard' (unconditional) with mask_xt=true makes no sense — "
            "the model has no conditioning channels to read."
        )

    if prediction_target == "x0" and config.get("loss_function") == "physical":
        issues.append(
            "PhysicalLossStrategy computes div(ε̂) but prediction_target='x0' predicts x₀, not ε. "
            "The divergence penalty would be applied to the wrong quantity."
        )

    return issues


def main():
    parser = argparse.ArgumentParser(
        description="Launch a DDPM inpainting experiment"
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to the experiment's config.yaml (overrides)",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=DEFAULT_TEMPLATE,
        help="Base template YAML to merge with (default: base_inpaint.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print resolved config without training",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Override epochs=3 for a quick smoke test",
    )
    args = parser.parse_args()

    # --- Load and merge configs ---
    if not args.template.exists():
        print(f"ERROR: template not found: {args.template}")
        sys.exit(1)
    if not args.config.exists():
        print(f"ERROR: config not found: {args.config}")
        sys.exit(1)

    with open(args.template) as f:
        base = yaml.safe_load(f) or {}
    with open(args.config) as f:
        overrides = yaml.safe_load(f) or {}

    resolved = deep_merge(base, overrides)

    if args.smoke:
        resolved["epochs"] = 3
        resolved["model_name"] = resolved.get("model_name", "smoke_test") + "_smoke"

    # --- Validate ---
    print("=" * 60)
    print("EXPERIMENT CONFIG VALIDATION")
    print("=" * 60)
    print(f"  Template:  {args.template}")
    print(f"  Overrides: {args.config}")
    print(f"  Noise:     {resolved.get('noise_function')}")
    print(f"  UNet:      {resolved.get('unet_type')}")
    print(f"  Predict:   {resolved.get('prediction_target')}")
    print(f"  Epochs:    {resolved.get('epochs')}")
    print(f"  Batch:     {resolved.get('batch_size')}")
    print()

    issues = validate_config(resolved)
    if issues:
        print("COMPATIBILITY ISSUES:")
        for issue in issues:
            print(f"  ✗ {issue}")
        print()
        if args.dry_run:
            print("--- Resolved config (WITH ISSUES) ---")
            print(yaml.dump(resolved, default_flow_style=False, sort_keys=False))
        else:
            print("Aborting. Fix the config or use --dry-run to inspect.")
        sys.exit(1)
    else:
        print("  ✓ All component compatibility checks passed")
    print()

    # --- Write resolved config ---
    exp_dir = args.config.resolve().parent
    results_dir = exp_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Point training outputs into the experiment's results/ folder
    resolved["output_dir"] = str(results_dir)

    resolved_path = results_dir / "resolved_config.yaml"
    with open(resolved_path, "w") as f:
        yaml.dump(resolved, f, default_flow_style=False, sort_keys=False)
    print(f"  Resolved config written to: {resolved_path}")
    print(f"  Training output dir:        {results_dir}")

    if args.dry_run:
        print()
        print("--- Resolved config ---")
        print(yaml.dump(resolved, default_flow_style=False, sort_keys=False))
        return

    # --- Launch training ---
    print()
    print("Launching training...")
    print("=" * 60)

    cmd = [
        sys.executable,
        str(BASE_DIR / "ddpm" / "training" / "train_inpaint.py"),
        "--training_cfg",
        str(resolved_path),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(BASE_DIR) + ":" + env.get("PYTHONPATH", "")

    result = subprocess.run(cmd, env=env, cwd=str(BASE_DIR))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
