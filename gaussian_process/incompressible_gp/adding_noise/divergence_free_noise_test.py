import torch
import matplotlib.pyplot as plt


def exact_div_free_field_from_stream(H=128, W=128, freq=4.0, device='cpu'):
    x = torch.linspace(0, 2 * torch.pi, W, device=device)
    y = torch.linspace(0, 2 * torch.pi, H, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    phase_x, phase_y = 2 * torch.pi * torch.rand(2, device=device)
    psi = torch.sin(freq * X + phase_x) * torch.sin(freq * Y + phase_y)

    vx = torch.zeros_like(psi)
    vy = torch.zeros_like(psi)

    # Forward difference for vx = dψ/dy
    vx[:, :-1] = psi[:, 1:] - psi[:, :-1]
    vx[:, -1] = 0

    # Forward difference for vy = -dψ/dx
    vy[:-1, :] = -(psi[1:, :] - psi[:-1, :])
    vy[-1, :] = 0

    return vx, vy


# Sum N random fields
N = 4
H, W = 128, 128
vx_total = torch.zeros(H, W)
vy_total = torch.zeros(H, W)

for _ in range(N):
    freq = torch.empty(1).uniform_(1.0, 8.0).item()  # random freq between 1 and 8
    vx, vy = exact_div_free_field_from_stream(H, W, freq)
    vx_total += vx
    vy_total += vy

# Visualize summed field
X, Y = torch.meshgrid(torch.linspace(0, 1, W), torch.linspace(0, 1, H), indexing='ij')

plt.figure(figsize=(6, 6))
plt.quiver(X[::4, ::4], Y[::4, ::4], vx_total[::4, ::4], vy_total[::4, ::4], scale=40, color='blue')
plt.title(f"Sum of {N} Divergence-Free Fields")
plt.axis('equal')
plt.grid(True)
plt.show()

vx, vy = exact_div_free_field_from_stream(H=128, W=128, freq=4.0)

total_vx = vx.sum()
total_vy = vy.sum()

print(f"Total vx: {total_vx.item():.6f}")
print(f"Total vy: {total_vy.item():.6f}")

