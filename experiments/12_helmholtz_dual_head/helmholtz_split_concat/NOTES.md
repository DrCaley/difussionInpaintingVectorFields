# helmholtz_split_concat — Experiment Notes

## Rationale

Palette-style concatenation conditioning: feed `[x_t, mask(1ch), voronoi_fill(2ch)]`
as 5 channels directly into the existing `MyUNet_Helmholtz_Split` encoder. No separate
conditioning pathway — all information flows through the convolutional backbone jointly.

This is the simplest possible conditioning approach and avoids the spatial information
bottleneck that plagued the original FiLM (AdaGN) implementation.

## Architecture

- Uses `MyUNet_Helmholtz_Split` with `in_channels=5` (no code changes to the UNet itself — 
  it already supports the parameter)
- Same dual-head Helmholtz decomposition (ψ for curl-free, φ for div-free)
- Same channel structure as spatial FiLM experiment, just different conditioning mechanism

## Controlled Variables (vs spatial FiLM experiment)

- Same dataset, same Voronoi forward process, same mask fractions
- Same hyperparameters: lr=0.0005, cosine schedule, warmup=10, batch_size=16
- Same Helmholtz loss weights: λ_decomp=0.1, λ_orth=0.01
- Same EMA settings: decay=0.9999
- Same prediction target: x0
- mask_xt=false (model sees full x_t signal)

## Varied Variable

- **Conditioning mechanism**: Palette concat (5ch input) vs Spatial FiLM (2ch input + per-pixel modulation)

---

## Log

