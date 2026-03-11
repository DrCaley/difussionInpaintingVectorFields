import torch, numpy as np, sys
sys.path.insert(0, '.')

vcnn = torch.load('results/voronoi_cnn_eval/voronoi_cnn_eval_results.pt', map_location='cpu', weights_only=False)
gp_eval = torch.load('results/eddy_balanced_eval/bulk_eval_eddy_balanced_100.pt', map_location='cpu', weights_only=False)

gp_by_vidx = {}
gpdiff_by_vidx = {}
for s in gp_eval['samples']:
    vi = s['val_idx']
    gp_by_vidx[vi] = s['gp_mse']
    gpdiff_by_vidx[vi] = s['ddpm_mse']

vcnn_beats_gp = vcnn_beats_gpdiff = matched_cnt = 0
for r in vcnn['results']:
    vi = r['val_idx']
    if vi in gp_by_vidx:
        matched_cnt += 1
        if r['mse_missing_only'] < gp_by_vidx[vi]:
            vcnn_beats_gp += 1
        if r['mse_missing_only'] < gpdiff_by_vidx[vi]:
            vcnn_beats_gpdiff += 1

print(f'WIN RATES (matched {matched_cnt}/100)')
print(f'  Voronoi-CNN beats GP:      {vcnn_beats_gp}/{matched_cnt} ({100*vcnn_beats_gp/matched_cnt:.0f}%)')
print(f'  Voronoi-CNN beats GP-Diff: {vcnn_beats_gpdiff}/{matched_cnt} ({100*vcnn_beats_gpdiff/matched_cnt:.0f}%)')
