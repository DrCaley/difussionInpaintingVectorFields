import torch
ckpt = torch.load('experiments/10_topology_metrics/voronoi_topo_training/results/inpaint_gaussian_t250_best_checkpoint.pt', map_location='cpu', weights_only=False)
print('Keys:', list(ckpt.keys()))
for k in ckpt:
    if k != 'model_state_dict' and k != 'ema_state_dict' and k != 'optimizer_state_dict':
        print(f'  {k}: {ckpt[k]}')
