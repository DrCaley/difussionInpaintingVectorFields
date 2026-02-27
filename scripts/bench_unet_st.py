"""Benchmark full MyUNet_ST forward+backward pass after Conv3d→Conv1d fix."""
import torch
import time
import sys
sys.path.insert(0, '.')
from ddpm.neural_networks.unets.unet_st import MyUNet_ST

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Device: {device}")

B, T = 4, 13

model = MyUNet_ST(n_steps=250, T=T).to(device)
x = torch.randn(B, T * 2, 64, 128, device=device)
t_step = torch.randint(0, 250, (B,), device=device)

print(f"=== Full forward pass (B={B}, T={T}) ===")
# Warmup
with torch.no_grad():
    for _ in range(2):
        y = model(x, t_step)
    torch.mps.synchronize()

# Timed forward only
with torch.no_grad():
    t0 = time.perf_counter()
    for _ in range(3):
        y = model(x, t_step)
        torch.mps.synchronize()
    t_fwd = (time.perf_counter() - t0) / 3

print(f"Forward pass:  {t_fwd * 1000:.0f} ms")
print(f"Output shape:  {y.shape}")

# Forward + backward (training mode)
model.train()
print(f"\n=== Forward + backward (B={B}, T={T}) ===")
# Warmup
y = model(x, t_step)
loss = y.sum()
loss.backward()
torch.mps.synchronize()

# Timed
t0 = time.perf_counter()
for _ in range(3):
    model.zero_grad()
    y = model(x, t_step)
    loss = y.sum()
    loss.backward()
    torch.mps.synchronize()
t_total = (time.perf_counter() - t0) / 3

print(f"Fwd+bwd:       {t_total * 1000:.0f} ms  (B={B})")
print(f"Est. at B=8:   {t_total * 2 * 1000:.0f} ms")
print(f"\nEst. epoch time at B=8, 7598 samples:")
batches_per_epoch = 7598 // 8
epoch_time_s = t_total * 2 * batches_per_epoch
print(f"  {batches_per_epoch} batches × {t_total*2*1000:.0f} ms = {epoch_time_s/60:.1f} min/epoch")
print(f"  500 epochs = {epoch_time_s * 500 / 3600:.1f} hours")
