# helmholtz_matched_fullsup_800

## 2026-03-10: Experiment Created

**Motivation**: The 400-epoch fullsup run proved that full-time decomposition
supervision fixes head degeneracy (cos_sol=0.95, cancel=1.3x) but at a ~10%
reconstruction cost (1.142x V-CNN vs 1.038x for matched noise without supervision).
The original matched noise model also showed signs of LR starvation — it was
still improving at epoch 328 when the cosine schedule decayed too aggressively.

This run gives the model 800 epochs with full supervision from scratch, allowing
the cosine schedule to decay more gradually and giving the model enough time to
optimize both reconstruction quality and head meaningfulness simultaneously.

**Config**: Same as helmholtz_matched_fullsup but with 800 epochs instead of 400.
- `lambda_decomp: 0.5`, `lambda_orth: 0.1`, `decomp_t_max_frac: 1.0`
- `lr: 0.0005`, cosine schedule, 10-epoch warmup
- Training from scratch (no init_from_weights)

**Server**: Server 2 (1.208.108.242:58777, RTX 3060)
