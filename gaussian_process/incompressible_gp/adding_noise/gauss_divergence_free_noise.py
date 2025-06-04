import torch
import matplotlib.pyplot as plt


def divergence_free_noise(H=128, W=128, freq=100000, device='cpu'):

    SEED = 0
    torch.random.manual_seed(SEED)

    # Create a scalar stream function ψ with sinusoidal content
    x = torch.linspace(0, 2 * torch.pi, W, device=device)
    y = torch.linspace(0, 2 * torch.pi, H, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    phase_x, phase_y = 2 * torch.pi * torch.rand(2)
    psi = torch.sin(freq * X + phase_x) * torch.sin(freq * Y + phase_y)

    # Create vx = dψ/dy, vy = -dψ/dx using forward differences
    vx = torch.zeros_like(psi)
    vy = torch.zeros_like(psi)

    # Use forward difference for vx
    vx[:, :-1] = psi[:, 1:] - psi[:, :-1]
    vx[:, -1] = 0  # Optional: zero padding

    # Use forward difference for vy
    vy[:-1, :] = -(psi[1:, :] - psi[:-1, :])
    vy[-1, :] = 0  # Optional: zero padding

    return vx, vy







vx, vy = divergence_free_noise(H=128, W=128, freq=2.0)

# Visualize
X, Y = torch.meshgrid(torch.linspace(0, 1, vx.shape[1]), torch.linspace(0, 1, vx.shape[0]), indexing='ij')
plt.figure(figsize=(6, 6))
plt.quiver(X[::1, ::1], Y[::1, ::1], vx[::1, ::1], vy[::1, ::1], scale=20)
plt.title("Exactly Divergence-Free Vector Field (Discrete)")
plt.axis('equal')
plt.grid(True)
plt.show()

vx, vy = divergence_free_noise(H=128, W=128, freq=4.0)

total_vx = vx.sum()
total_vy = vy.sum()

print(f"Total vx: {total_vx.item():.6f}")
print(f"Total vy: {total_vy.item():.6f}")

