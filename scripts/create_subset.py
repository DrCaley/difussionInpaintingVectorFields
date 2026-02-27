import pickle
import numpy as np
import h5py

def create_subset(
    input_file: str,
    output_file: str,
    target_lat: int = 44,
    target_lon: int = 94,
    row_start: int = None,
    col_start: int = None
):
    """
    Load a MATLAB v7.3 HDF5 .mat file (242 x 329), extract a spatial subset,
    and save it as a pickle file.

    Args:
        input_file:  Path to the input .mat file.
        output_file: Path to save the subsetted pickle file.
        target_lat:  Number of latitude points in the subset (default 44).
        target_lon:  Number of longitude points in the subset (default 94).
        row_start:   Starting row index (default: center of domain).
        col_start:   Starting column index (default: center of domain).
    """

    # --- Load using h5py, same as load_h5() in dataset_visualizer ---
    print(f"Loading dataset from: {input_file}")
    with h5py.File(input_file, "r") as f:
        print("Keys:", list(f.keys()))
        u_raw   = np.array(f['us'])       # (T, H, W)
        v_raw   = np.array(f['vs'])
        lat     = np.array(f['lat_rho'])  # (H, W)
        lon     = np.array(f['lon_rho'])

    # Keep raw (T, H, W) order — transposing is done in the visualizer's load_pickle()
    T, H, W = u_raw.shape
    print(f"Loaded shape: {H} x {W} x {T}")

    # Validate target size
    if target_lat > H or target_lon > W:
        raise ValueError(
            f"Target size ({target_lat}x{target_lon}) exceeds "
            f"source size ({H}x{W})."
        )

    # Default: centre the subset in the domain
    if row_start is None:
        row_start = (H - target_lat) // 2
    if col_start is None:
        col_start = (W - target_lon) // 2

    row_end = row_start + target_lat
    col_end = col_start + target_lon
    print(f"Subsetting rows[{row_start}:{row_end}], cols[{col_start}:{col_end}]")

    # --- Slice spatial dims (H, W), keep all time steps ---
    subset = {
        'us':      u_raw[:, row_start:row_end, col_start:col_end],   # (T, target_lat, target_lon)
        'vs':      v_raw[:, row_start:row_end, col_start:col_end],
        'lat_rho': lat[row_start:row_end, col_start:col_end],        # (target_lat, target_lon)
        'lon_rho': lon[row_start:row_end, col_start:col_end],
    }

    for k, arr in subset.items():
        print(f"  '{k}': {arr.shape}")

    # --- Save as pickle ---
    with open(output_file, 'wb') as f:
        pickle.dump(subset, f)
    print(f"\nSubset saved to: {output_file}")


# ── Example usage ──────────────────────────────────────────────────────────────
if __name__ == "__main__":

    INPUT_FILE  = "data/stjohn_hourly_surface_velocity_20250718.mat"
    OUTPUT_FILE = "data/stjohn_subset_ocean.pkl"

    create_subset(
        input_file  = INPUT_FILE,
        output_file = OUTPUT_FILE,
        target_lat  = 44,
        target_lon  = 94,
        # Set row_start / col_start to choose a specific region,
        # or leave as None to auto-centre the subset.
        row_start   = 0,
        col_start   = 0,
    )