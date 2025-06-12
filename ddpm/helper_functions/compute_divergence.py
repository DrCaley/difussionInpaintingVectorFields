import torch

def compute_divergence(vx, vy):
    """
    Computes the discrete divergence of a 2D vector field at every point.
    Uses central differences (zero padding at the borders).
    """
    H, W = vx.shape

    # Initialize divergence
    divergence = torch.zeros_like(vx)

    # Central difference for interior points
    dvx_dx = torch.zeros_like(vx)
    dvy_dy = torch.zeros_like(vy)

    dvx_dx[1:-1, :] = (vx[2:, :] - vx[:-2, :]) / 2.0
    dvy_dy[:, 1:-1] = (vy[:, 2:] - vy[:, :-2]) / 2.0

    divergence = dvx_dx + dvy_dy
    return divergence

"""
After a few experiments, I conclude that this is equivalent to this function. -Matt

# Grid setup
Nx, Ny = 64, 64
Lx, Ly = 2 * np.pi, 2 * np.pi
dx, dy = Lx / Nx, Ly / Ny

# Another divergence to use
def compute_divergence1(u, v, dx):
    dudx = np.gradient(u, dx, axis=0)
    dvdy = np.gradient(v, dx, axis=1)
    return dudx + dvdy
"""
