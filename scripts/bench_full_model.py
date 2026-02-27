"""Benchmark full MyUNet_ST forward+backward after Conv1d fix."""
import torch
import time
import sys
sys.path.insert(0, '.')
from ddpm.neural_networks.unets.unet_st import MyUNet_ST

device = torch.device('mps')

for B in [2, 4, 8]:
    T = 13
    model = MyUNet_ST(n_steps=250, T=T).to(device)
    x = torch.randn(B, T * 2, 64, 128, device=device)
    t_step = torch.randint(0, 250, (B,), device=device)

    # Warmup
    model.eval()
    with torch.no_grad():
        y = model(x, t_step)
        torch.mps.synchronize()

    # Forward only
    with torch.no_grad():
        t0 = time.perf_counter()
        y = model(x, t_step)
        torch.mps.synchronize()
    t_fwd = (time.perf_counter() - t0) * 1000

    # Forward + backward
    model.train()
    model.zero_grad()
    y = model(x, t_step)
    y.sum().backward()
    torch.mps.synchronize()

    t0 = time.perf_counter()
    for _ in range(2):
        model.zero_grad()
        y = model(x, t_step)
        y.sum().backward()
        torch.mps.synchronize()
    t_fb = (time.perf_counter() - t0) / 2 * 1000

    n_batches = 7598 // B
    epoch_min = t_fb / 1000 * n_batches / 60
    print(f"B={B}: fwd={t_fwd:.0f}ms  fwd+bwd={t_fb:.0f}ms  "
          f"batches/ep={n_batches}  epoch={epoch_min:.1f}min  "
          f"500ep={epoch_min*500/60:.0f}h")
    del model, x, y
    torch.mps.empty_cache()
