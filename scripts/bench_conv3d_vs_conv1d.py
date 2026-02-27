"""Benchmark Conv3d (k,1,1) vs Conv1d on MPS to identify the bottleneck."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

B, T, C, H, W = 8, 13, 64, 64, 128  # Level-1 dimensions

# ─── Conv3d with kernel (3,1,1) ───
conv3d = nn.Conv3d(C, C, kernel_size=(3,1,1), padding=(1,0,0)).to(device)
x_5d = torch.randn(B, C, T, H, W, device=device)

# Warmup
for _ in range(3):
    _ = conv3d(x_5d)
    torch.mps.synchronize()

t0 = time.perf_counter()
for _ in range(10):
    _ = conv3d(x_5d)
    torch.mps.synchronize()
t_conv3d = (time.perf_counter() - t0) / 10
print(f"Conv3d (B={B}, C={C}, T={T}, H={H}, W={W}):  {t_conv3d*1000:.1f} ms")

# ─── Conv1d equivalent ───
conv1d = nn.Conv1d(C, C, kernel_size=3, padding=1).to(device)
# Copy weights for fairness
with torch.no_grad():
    conv1d.weight.copy_(conv3d.weight.squeeze(-1).squeeze(-1))
    conv1d.bias.copy_(conv3d.bias)

x_3d = x_5d.permute(0, 3, 4, 1, 2).reshape(B*H*W, C, T)  # (B*H*W, C, T)

for _ in range(3):
    _ = conv1d(x_3d)
    torch.mps.synchronize()

t0 = time.perf_counter()
for _ in range(10):
    _ = conv1d(x_3d)
    torch.mps.synchronize()
t_conv1d = (time.perf_counter() - t0) / 10
print(f"Conv1d (B*H*W={B*H*W}, C={C}, T={T}):       {t_conv1d*1000:.1f} ms")

print(f"\nSpeedup: {t_conv3d/t_conv1d:.1f}x")

# ─── Also test reshape + permute overhead ───
x_4d = torch.randn(B*T, C, H, W, device=device)

for _ in range(3):
    h = x_4d.reshape(B, T, C, H, W).permute(0, 3, 4, 2, 1).reshape(B*H*W, C, T)
    torch.mps.synchronize()

t0 = time.perf_counter()
for _ in range(10):
    h = x_4d.reshape(B, T, C, H, W).permute(0, 3, 4, 2, 1).reshape(B*H*W, C, T)
    torch.mps.synchronize()
t_reshape = (time.perf_counter() - t0) / 10
print(f"\nReshape (B*T,C,H,W)→(B*H*W,C,T):           {t_reshape*1000:.1f} ms")

# ─── Full forward pass timing: Conv3d path vs Conv1d path ───
print("\n─── Full temporal conv block simulation ───")

# Conv3d path (current)
def conv3d_path(x_4d, conv3d, B, T):
    BT, C, H, W = x_4d.shape
    h = x_4d.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)
    h = F.silu(conv3d(h))
    h = h.permute(0, 2, 1, 3, 4).reshape(BT, C, H, W)
    return x_4d + h

# Conv1d path (proposed)
def conv1d_path(x_4d, conv1d, B, T):
    BT, C, H, W = x_4d.shape
    h = x_4d.reshape(B, T, C, H, W).permute(0, 3, 4, 2, 1).reshape(B*H*W, C, T)
    h = F.silu(conv1d(h))
    h = h.reshape(B, H, W, C, T).permute(0, 4, 3, 1, 2).reshape(BT, C, H, W)
    return x_4d + h

for _ in range(3):
    _ = conv3d_path(x_4d, conv3d, B, T)
    _ = conv1d_path(x_4d, conv1d, B, T)
    torch.mps.synchronize()

t0 = time.perf_counter()
for _ in range(10):
    _ = conv3d_path(x_4d, conv3d, B, T)
    torch.mps.synchronize()
t_path3d = (time.perf_counter() - t0) / 10

t0 = time.perf_counter()
for _ in range(10):
    _ = conv1d_path(x_4d, conv1d, B, T)
    torch.mps.synchronize()
t_path1d = (time.perf_counter() - t0) / 10

print(f"Conv3d full path: {t_path3d*1000:.1f} ms")
print(f"Conv1d full path: {t_path1d*1000:.1f} ms")
print(f"Speedup:          {t_path3d/t_path1d:.1f}x")

# ─── Test scaled_dot_product_attention on MPS ───
print("\n─── Temporal attention benchmark ───")
BHW = B * 16 * 32  # Level 3 spatial resolution
n_heads, d_k = 4, 64
q = torch.randn(BHW, n_heads, T, d_k, device=device)
k = torch.randn(BHW, n_heads, T, d_k, device=device)
v = torch.randn(BHW, n_heads, T, d_k, device=device)

for _ in range(3):
    _ = F.scaled_dot_product_attention(q, k, v)
    torch.mps.synchronize()

t0 = time.perf_counter()
for _ in range(10):
    _ = F.scaled_dot_product_attention(q, k, v)
    torch.mps.synchronize()
t_sdpa = (time.perf_counter() - t0) / 10
print(f"SDPA (B*H*W={BHW}, heads={n_heads}, T={T}, d_k={d_k}): {t_sdpa*1000:.1f} ms")
