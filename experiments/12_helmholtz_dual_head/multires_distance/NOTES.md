# multires_distance — Experiment Notes

## Hypothesis

Adding a normalized distance-to-nearest-sensor field (V-CNN inspired) as an
extra conditioning channel gives the denoiser explicit spatial awareness of how
far each pixel is from the nearest observation. This should help the model
allocate uncertainty more effectively, especially at very sparse coverage where
the density channel in the multi-res encoder may not provide enough geometric
information.

## Changes from multires_splitnoise (control)

- **New config key**: `use_distance_field: true`
- **Dataset**: `OceanInpaintDataset` computes EDT from mask, normalizes to [0,1],
  appends to `known_values` → 3ch instead of 2ch
- **UNet input**: 6 channels (x_t[2] + mask[1] + obs_u[1] + obs_v[1] + dist[1])
  instead of 5
- **MultiResCondEncoder**: Accepts 4ch cond; distance field pooled with
  `avg_pool2d` (dense, not sparse) and concatenated at each FPN level
- **Everything else identical**: same split noise, same loss, same hyperparams

## Run Log
