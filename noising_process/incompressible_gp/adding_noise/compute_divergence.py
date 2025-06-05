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
