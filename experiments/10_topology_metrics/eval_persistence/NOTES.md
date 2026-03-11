# Eval Persistence — Lab Notebook

## Goal

Evaluate existing GP-Diff reconstructions with multi-scalar persistent
homology to measure topological fidelity beyond pixel-wise MSE.

## Status: Not started

---

## TODO

1. [ ] Verify gudhi installed and sanity_check.py passes
2. [ ] Run persistence_metrics.py on 100-sample bulk eval
   ```bash
   PYTHONPATH=. python experiments/10_topology_metrics/persistence_metrics.py \
     --results-pt experiments/08_network_architecture/repaint_gaussian_attn/results/bulk_eval_best_100samples.pt \
     --output experiments/10_topology_metrics/eval_persistence/results/persistence_100.pt \
     --format bulk-eval-list
   ```
3. [ ] Compare W2 distances: GP vs GP-Diff across all 4 scalar fields
4. [ ] Check topology metric correlation with Γ₁ F1 scores
5. [ ] Sensitivity analysis: vary σ (0.5, 1.0, 2.0) and weights
6. [ ] Run on full validation set if results look promising
7. [ ] Visualize persistence diagrams for best/worst samples

## Entries

*(Add dated entries below as work progresses)*
