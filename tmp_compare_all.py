#!/usr/bin/env python3
"""Compare all methods side by side."""
import torch, numpy as np, sys
sys.path.insert(0, '.')

# GP-Diff (unconditional DDPM) — 100 samples
gp_eval = torch.load('results/eddy_balanced_eval/bulk_eval_eddy_balanced_100.pt',
                     map_location='cpu', weights_only=False)
samples = gp_eval['samples']
gp_mses = [s['gp_mse'] for s in samples]
gpdiff_mses = [s['ddpm_mse'] for s in samples]

# Voronoi-CNN — 100 samples
vcnn = torch.load('results/voronoi_cnn_eval/voronoi_cnn_eval_results.pt',
                  map_location='cpu', weights_only=False)
vcnn_mses = [r['mse_missing_only'] for r in vcnn['results']]

# FiLM+GP eval — 20 samples
fg = torch.load('results/film_attn_gp_eval/film_attn_gp_eval.pt',
               map_location='cpu', weights_only=False)
fg_results = fg.get('results', fg.get('samples', []))
fg_gp = [r['gp_mse'] for r in fg_results]
fg_film = [r.get('film_attn_mse', r.get('film_mse', r.get('ddpm_mse', 0))) for r in fg_results]

# FiLM attn eval — 100 samples
fa = torch.load('results/film_attn_eval/film_attn_eval_100.pt',
               map_location='cpu', weights_only=False)
fa_results = fa.get('results', fa.get('samples', []))
print(f"FiLM-attn 100 keys: {list(fa.keys())}")
if fa_results:
    print(f"  Sample keys: {list(fa_results[0].keys())}")
    fa_gp = [r.get('gp_mse', 0) for r in fa_results]
    fa_film = [r.get('film_mse', r.get('ddpm_mse', 0)) for r in fa_results]

# FiLM conditional eval
fc = torch.load('results/film_conditional_eval/film_conditional_eval_100.pt',
               map_location='cpu', weights_only=False)
fc_results = fc.get('results', fc.get('samples', []))
print(f"FiLM-cond 100 keys: {list(fc.keys())}")
if fc_results:
    print(f"  Sample keys: {list(fc_results[0].keys())}")
    fc_gp = [r.get('gp_mse', 0) for r in fc_results]
    fc_film = [r.get('film_mse', r.get('ddpm_mse', 0)) for r in fc_results]

print("\n" + "="*70)
print("ALL METHODS COMPARISON (mean MSE, missing pixels)")
print("="*70)
print(f"  GP baseline:       {np.mean(gp_mses):.6f}  (100 samples)")
print(f"  Voronoi-CNN:       {np.mean(vcnn_mses):.6f}  (100 samples)  ratio={np.mean(vcnn_mses)/np.mean(gp_mses):.3f}")
print(f"  GP-Diff (uncond):  {np.mean(gpdiff_mses):.6f}  (100 samples)  ratio={np.mean(gpdiff_mses)/np.mean(gp_mses):.3f}")
if fa_results:
    print(f"  FiLM-attn (100):   {np.mean(fa_film):.6f}  (100 samples)  ratio={np.mean(fa_film)/np.mean(fa_gp):.3f}")
if fc_results:
    print(f"  FiLM-cond (100):   {np.mean(fc_film):.6f}  (100 samples)  ratio={np.mean(fc_film)/np.mean(fc_gp):.3f}")
print(f"  FiLM+GP (20):      {np.mean(fg_film):.6f}  (20 samples)   ratio={np.mean(fg_film)/np.mean(fg_gp):.3f}")
print(f"    FiLM+GP GP base: {np.mean(fg_gp):.6f}")

# Check which epoch/weights the FiLM evals used
print(f"\nFiLM+GP config: {fg.get('config', 'N/A')}")
print(f"FiLM-attn config: {fa.get('config', fa.get('hyperparams', 'N/A'))}")
print(f"FiLM-cond config: {fc.get('config', fc.get('hyperparams', 'N/A'))}")
