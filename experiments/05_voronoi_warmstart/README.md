# 05 — Voronoi Warm-Start for Diffusion Inpainting

## Research Question

Can we replace the GP posterior mean with a Voronoi tessellation as the
warm-start for RePaint diffusion inpainting, and use distance-from-sensor
as a variance proxy instead of the GP posterior variance?

**Motivation**: The Voronoi-CNN baseline (Fukami et al. 2021) achieves
3× lower MSE than GP-Diff despite using a simple U-Net. The key insight
from that work is that Voronoi tessellation is a powerful, zero-cost
preprocessing step that gives CNNs a dense, structured input. We test
whether feeding the same Voronoi-tessellated field into the diffusion
pipeline (instead of the GP mean) improves reconstruction quality.

## Controlled Variables
- Same unconditional DDPM checkpoint (repaint_gaussian_attn)
- Same 100-sample eddy-balanced evaluation set
- Same S6 adaptive RePaint inference loop
- Same Gamma-1 eddy detection parameters

## Varied Variables
- **Prior image**: GP posterior mean → Voronoi tessellation (nearest-neighbour fill)
- **Variance map**: GP posterior variance → distance-from-sensor (squared, normalised)

## Experiments

### `voronoi_gp_replace/`
Direct swap: Voronoi fill replaces GP mean, distance² replaces GP variance.
Everything else identical to the production GP-Diff pipeline.
