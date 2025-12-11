import torch
import torch.nn.functional as F

def interpolate_masked_velocity_field(velocity: torch.Tensor, mask: torch.Tensor, max_iters: int = 2000) -> torch.Tensor:
    """
    PyTorch-only safe interpolation of a masked 2-channel velocity field.
    Ensures all masked regions are filled using neighbor propagation.

    Args:
        velocity (torch.Tensor): shape (2, H, W)
        mask (torch.Tensor): shape (1, H, W), 1 = missing, 0 = known
        max_iters (int): Max iterations to propagate values

    Returns:
        torch.Tensor: Filled velocity of shape (2, H, W)
    """
    assert velocity.shape[0] == 2 and mask.shape[0] == 1
    device = velocity.device
    kernel = torch.ones(1, 1, 3, 3, device=device)
    padding = 1

    known_mask = (1 - mask).clone()  # 1 = known, 0 = unknown
    filled = velocity.clone()

    for _ in range(max_iters):
        still_missing = (known_mask == 0)

        if still_missing.sum() == 0:
            break  # done, all filled

        for c in range(2):  # for both velocity components
            v = filled[c:c+1]  # shape (1, H, W)

            # Sum of known neighbors
            neighbor_sum = F.conv2d(v * known_mask, kernel, padding=padding)
            neighbor_count = F.conv2d(known_mask, kernel, padding=padding)

            # Avoid divide-by-zero
            avg_neighbors = neighbor_sum / (neighbor_count + 1e-6)

            # Only update unknowns
            v_new = torch.where(still_missing, avg_neighbors, v)
            filled[c:c+1] = v_new

        # Update known mask to include newly filled pixels
        known_mask = torch.where(still_missing & (neighbor_count > 0), torch.tensor(1.0, device=device), known_mask)

    return filled

def rbf_kernel(X1, X2, lengthscale=1.0, variance=1.0):
    """
    Compute the RBF (Gaussian) kernel between two sets of points.
    """
    dist_sq = torch.cdist(X1, X2).pow(2)
    return variance * torch.exp(-0.5 * dist_sq / lengthscale**2)

def gp_fill(tensor, mask, lengthscale=1.5, variance=1.0, noise=1e-6, use_double=True):
    dtype = torch.float64 if use_double else tensor.dtype
    device = tensor.device

    tensor = tensor.to(dtype)
    mask = mask.to(dtype)

    _, C, H, W = tensor.shape
    filled = tensor.clone()

    # Normalize factor
    norm_factor = torch.tensor([H, W], dtype=dtype, device=device)

    for c in range(C):
        channel_data = tensor[0, c]
        channel_mask = mask[0, c]

        known_idx = (channel_mask == 0).nonzero()
        unknown_idx = (channel_mask == 1).nonzero()

        if known_idx.numel() == 0 or unknown_idx.numel() == 0:
            continue

        known_xy = known_idx.float() / norm_factor
        unknown_xy = unknown_idx.float() / norm_factor

        y = channel_data[known_idx[:, 0], known_idx[:, 1]]

        K = rbf_kernel(known_xy, known_xy, lengthscale, variance)
        K_s = rbf_kernel(unknown_xy, known_xy, lengthscale, variance)
        K_ss = rbf_kernel(unknown_xy, unknown_xy, lengthscale, variance)

        # Cholesky solve
        jitter = max(noise, 1e-6)
        for _ in range(5):
            try:
                L = torch.linalg.cholesky(K + jitter * torch.eye(K.shape[0], device=device, dtype=dtype))
                break
            except RuntimeError:
                jitter *= 10

        alpha = torch.cholesky_solve(y.unsqueeze(-1), L)
        pred_mean = (K_s @ alpha).squeeze()

        # Fill in one batched operation
        filled[0, c][unknown_idx[:,0], unknown_idx[:,1]] = pred_mean

    return filled.to(torch.float32)
