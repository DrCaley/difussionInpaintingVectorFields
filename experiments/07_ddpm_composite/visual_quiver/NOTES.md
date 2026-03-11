# DDPM Composite — Visual Quiver Notes

## 2026-03-02: Initial setup

Migrated from `scripts/generate_single_panels.py` into experiment structure.

### Configuration
- **Coverage**: 1% (38 observation points out of ~3800 ocean cells)
- **Validation sample**: val[460]
- **DDPM timestep**: t=200 (optimal for 1% coverage)
- **Ensemble**: 5 members
- **Arrow style**: project standard (`plot_vector_field`), target_median_arrow_len=0.45, step=2

### Run command
```bash
PYTHONPATH=. python experiments/07_method_comparison/visual_quiver/run.py \
    --reveal-pct 1.0 --val-idx 460 --t-val 200
```

### Observations
- At 1% coverage, GP produces smooth but featureless fields (ℓ=14.1 >> point spacing)
- VCNN recovers eddy-scale structure despite only seeing Voronoi tessellation
- Composite blends GP-CNN (near observations) with DDPM (far from observations)
  via GP posterior variance weighting
- VCNN visually outperforms composite at extreme sparsity — consistent with
  eddy detection F1 scores (0.565 vs 0.450 at 1%)

### Previous results from `scripts/generate_single_panels.py`
Figures were previously output to `paper/figures/fig_{gt,gp,vcnn,composite}.png`.
Those are now superseded by the results in this experiment folder.
