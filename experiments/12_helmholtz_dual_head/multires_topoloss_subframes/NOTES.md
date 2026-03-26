# multires_topoloss_subframes

## What
Multires FiLM + Helmholtz split noise + topology penalties (vorticity + divergence)
trained on the subframes dataset with bathymetry conditioning.

Combines `multires_splitnoise_topo` architecture with `data_subframes` dataset.

## Controlled
- Same architecture, split noise, Helmholtz losses as multires_splitnoise_subframes
- Same data_subframes dataset + bathymetry

## Varied
- Added topology penalties: lambda_vort=0.01, lambda_div=0.005, t_max_frac=0.1

## Log

### 2026-03-25 — Initial training
- Training on server1 (RTX 3060, 12GB), batch_size=32
- Fresh start (no resume)
