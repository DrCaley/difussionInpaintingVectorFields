"""Isolate which component is slow on MPS."""
import torch
import torch.nn.functional as F
import time
import sys
sys.path.insert(0, '.')

device = torch.device('mps')
print(f"Device: {device}")

B, T = 4, 13

# ─── Test 1: Spatial UNet alone (B*T frames, no temporal) ───
print("\n=== Test 1: Spatial UNet (B*T=52 frames) ===")
from ddpm.neural_networks.unets.unet_xl_attn import MyUNet_Attn
spatial = MyUNet_Attn(n_steps=250).to(device)
x_spatial = torch.randn(B * T, 2, 64, 128, device=device)
t_step = torch.randint(0, 250, (B * T,), device=device)

with torch.no_grad():
    y = spatial(x_spatial, t_step)
    torch.mps.synchronize()
print(f"  Warmup done, shape: {y.shape}")

with torch.no_grad():
    t0 = time.perf_counter()
    y = spatial(x_spatial, t_step)
    torch.mps.synchronize()
print(f"  Forward: {(time.perf_counter()-t0)*1000:.0f} ms")

del spatial, x_spatial, y
torch.mps.empty_cache()

# ─── Test 2: TemporalConvBlock alone at each level ───
print("\n=== Test 2: TemporalConvBlock at each UNet level ===")
from ddpm.neural_networks.unets.unet_st import TemporalConvBlock

levels = [(64, 64, 128), (128, 32, 64), (256, 16, 32), (256, 8, 16), (256, 4, 8)]
for C, H, W in levels:
    block = TemporalConvBlock(C).to(device)
    x = torch.randn(B * T, C, H, W, device=device)
    with torch.no_grad():
        y = block(x, B, T)
        torch.mps.synchronize()
    with torch.no_grad():
        t0 = time.perf_counter()
        y = block(x, B, T)
        torch.mps.synchronize()
    print(f"  Level (C={C}, {H}x{W}): {(time.perf_counter()-t0)*1000:.1f} ms")
    del block, x, y

torch.mps.empty_cache()

# ─── Test 3: TemporalAttnBlock alone at each level ───
print("\n=== Test 3: TemporalAttnBlock at attention levels ===")
from ddpm.neural_networks.unets.unet_st import TemporalAttnBlock

attn_levels = [(256, 16, 32), (256, 8, 16), (256, 4, 8)]
for C, H, W in attn_levels:
    block = TemporalAttnBlock(C, T=T).to(device)
    x = torch.randn(B * T, C, H, W, device=device)
    with torch.no_grad():
        y = block(x, B, T)
        torch.mps.synchronize()
    with torch.no_grad():
        t0 = time.perf_counter()
        y = block(x, B, T)
        torch.mps.synchronize()
    dt = (time.perf_counter() - t0) * 1000
    print(f"  Level (C={C}, {H}x{W}): {dt:.1f} ms  [B*H*W={B*H*W}]")
    del block, x, y

torch.mps.empty_cache()

# ─── Test 4: scaled_dot_product_attention alone ───
print("\n=== Test 4: scaled_dot_product_attention ===")
for H, W in [(4, 8), (8, 16), (16, 32)]:
    BHW = B * H * W
    n_heads, d_k = 4, 64
    q = torch.randn(BHW, n_heads, T, d_k, device=device)
    k = torch.randn(BHW, n_heads, T, d_k, device=device)
    v = torch.randn(BHW, n_heads, T, d_k, device=device)
    with torch.no_grad():
        out = F.scaled_dot_product_attention(q, k, v)
        torch.mps.synchronize()
    with torch.no_grad():
        t0 = time.perf_counter()
        out = F.scaled_dot_product_attention(q, k, v)
        torch.mps.synchronize()
    dt = (time.perf_counter() - t0) * 1000
    print(f"  SDPA (B*H*W={BHW}, heads={n_heads}, T={T}): {dt:.1f} ms")
    del q, k, v, out

print("\nAll tests complete!")
