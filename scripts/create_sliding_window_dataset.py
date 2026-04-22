import numpy as np
import h5py


def create_sliding_window_dataset(
    input_file: str,
    output_file: str,
    target_lat: int = 44,
    target_lon: int = 94,
    stride_lat: int = None,
    stride_lon: int = None,
    min_ocean_fraction: float = 0.1,
):
    """
    Extract ALL spatial patches via a sliding window over the full domain and
    write them into an HDF5 file.  Patches are streamed one at a time — peak
    RAM is one patch × T, regardless of how many patches or timesteps there are.

    Output HDF5 keys:
        us       : (N_patches * T, target_lat, target_lon)
        vs       : (N_patches * T, target_lat, target_lon)
        lat_rho  : (target_lat, target_lon)  — from the first kept patch
        lon_rho  : (target_lat, target_lon)

    The output is readable by dataset_visualizer.load_h5() and by the
    '.h5' branch added to spliting_data_sets.load_dataset().

    Args:
        input_file:          Path to the input MATLAB v7.3 HDF5 .mat file.
        output_file:         Path for the output HDF5 file (.h5 / .hdf5).
        target_lat:          Patch height in grid points (default 44).
        target_lon:          Patch width  in grid points (default 94).
        stride_lat:          Row stride (default: target_lat // 2).
        stride_lon:          Col stride (default: target_lon // 2).
        min_ocean_fraction:  Fraction [0, 1] of non-NaN pixels a patch must
                             contain to be kept.  Default 0.1.
    """

    # ── Open source lazily — never load the full array into RAM ──────────────
    print(f"Opening dataset: {input_file}")
    with h5py.File(input_file, "r") as src, h5py.File(output_file, "w") as dst:
        print("Source keys:", list(src.keys()))

        us_src = src['us']                      # h5py dataset (T, H, W), not loaded
        vs_src = src['vs']
        lat    = np.array(src['lat_rho'])       # (H, W) — small, safe to load
        lon    = np.array(src['lon_rho'])

        T, H, W = us_src.shape
        dtype   = us_src.dtype
        print(f"Source shape: T={T}, H={H}, W={W}, dtype={dtype}")

        if target_lat > H or target_lon > W:
            raise ValueError(
                f"Patch size ({target_lat}×{target_lon}) exceeds domain ({H}×{W})."
            )

        # ── Sliding-window grid ───────────────────────────────────────────────
        if stride_lat is None:
            stride_lat = target_lat // 2
        if stride_lon is None:
            stride_lon = target_lon // 2

        row_starts = list(range(0, H - target_lat + 1, stride_lat))
        col_starts = list(range(0, W - target_lon + 1, stride_lon))
        if row_starts[-1] + target_lat < H:
            row_starts.append(H - target_lat)
        if col_starts[-1] + target_lon < W:
            col_starts.append(W - target_lon)

        n_positions = len(row_starts) * len(col_starts)
        print(
            f"Stride: ({stride_lat}, {stride_lon})  →  "
            f"{len(row_starts)} row × {len(col_starts)} col = {n_positions} candidates"
        )

        # ── Pass 1: identify kept patches (reads only t=0, very cheap) ────────
        kept_patches: list[tuple[int, int]] = []
        n_skipped = 0
        for r in row_starts:
            for c in col_starts:
                t0_u = us_src[0, r:r + target_lat, c:c + target_lon]
                ocean_fraction = float(np.sum(np.isfinite(t0_u))) / (target_lat * target_lon)
                if ocean_fraction < min_ocean_fraction:
                    n_skipped += 1
                else:
                    kept_patches.append((r, c))

        n_kept = len(kept_patches)
        print(
            f"Kept {n_kept} / {n_positions} patches  "
            f"(skipped {n_skipped} with ocean_fraction < {min_ocean_fraction})"
        )
        if n_kept == 0:
            raise RuntimeError(
                "No patches passed the ocean-fraction filter. "
                "Lower min_ocean_fraction or check the input data."
            )

        # ── Create resizable output datasets (chunked, written incrementally) ─
        chunk_t = min(256, T)
        us_dst = dst.create_dataset(
            'us', shape=(0, target_lat, target_lon),
            maxshape=(None, target_lat, target_lon),
            dtype=dtype, chunks=(chunk_t, target_lat, target_lon),
        )
        vs_dst = dst.create_dataset(
            'vs', shape=(0, target_lat, target_lon),
            maxshape=(None, target_lat, target_lon),
            dtype=dtype, chunks=(chunk_t, target_lat, target_lon),
        )

        # ── Pass 2: stream patches from source → output, one at a time ────────
        first_lat = first_lon = None
        offset = 0
        for idx, (r, c) in enumerate(kept_patches):
            print(
                f"  Writing patch {idx + 1}/{n_kept}  "
                f"rows {r}:{r + target_lat}, cols {c}:{c + target_lon}"
            )
            u_patch = us_src[:, r:r + target_lat, c:c + target_lon]   # (T, lat, lon)
            v_patch = vs_src[:, r:r + target_lat, c:c + target_lon]

            new_offset = offset + T
            us_dst.resize(new_offset, axis=0)
            vs_dst.resize(new_offset, axis=0)
            us_dst[offset:new_offset] = u_patch
            vs_dst[offset:new_offset] = v_patch
            offset = new_offset

            if first_lat is None:
                first_lat = lat[r:r + target_lat, c:c + target_lon]
                first_lon = lon[r:r + target_lat, c:c + target_lon]

        # ── Metadata ──────────────────────────────────────────────────────────
        dst.create_dataset('lat_rho', data=first_lat)
        dst.create_dataset('lon_rho', data=first_lon)
        dst.create_dataset('patch_origins', data=np.array(kept_patches))
        dst.attrs['n_patches']    = n_kept
        dst.attrs['T_per_patch']  = T
        dst.attrs['patch_size']   = [target_lat, target_lon]
        dst.attrs['stride']       = [stride_lat, stride_lon]
        dst.attrs['source_shape'] = [T, H, W]

        total = n_kept * T
        print(f"\nDone. {n_kept} patches × {T} timesteps = {total} total frames")
        print(f"Output: {output_file}  ({total} × {target_lat} × {target_lon})")


# ── Example usage ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    INPUT_FILE  = "data/stjohn_hourly_surface_velocity_20250718.mat"
    OUTPUT_FILE = "data/stjohn_varying_location.h5"

    create_sliding_window_dataset(
        input_file         = INPUT_FILE,
        output_file        = OUTPUT_FILE,
        target_lat         = 44,
        target_lon         = 94,
        # stride defaults to half the patch size (50 % overlap).
        # Set stride_lat=44 / stride_lon=94 for non-overlapping tiles.
        stride_lat         = None,
        stride_lon         = None,
        min_ocean_fraction = 0.6,
    )