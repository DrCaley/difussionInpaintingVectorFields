# multires_splitnoise_subframes — Experiment Notes

## Goal
Train the proven multires_splitnoise architecture on the expanded St. John
subframe dataset (29,400 samples, 3× more spatial diversity) with bathymetry
as an additional dense conditioning channel.

## What changed vs multires_splitnoise
- **Data**: subframes mmap dataset instead of original rams_head pickle
  (29,400 train samples vs ~9,800)
- **Bathymetry**: added as extra dense channel in MultiResCondEncoder
  (normalized [0,1] per subframe, pooled at each FPN level)
- **Data stats**: updated to match subframe dataset distribution

## Architecture
- MyUNet_Helmholtz_Split_FiLM_MultiRes with 6ch input
  [x_t(2), mask(1), sparse_u(1), sparse_v(1), bathy(1)]
- MultiResCondEncoder: pool_ch=4 (3 sparse + 1 bathymetry dense)

## Log

### 2026-03-24 — Initial launch
- Created config and deployed to server2
- Code changes: added `use_bathymetry` to MultiResCondEncoder,
  OceanInpaintDataset, and train_inpaint.py
