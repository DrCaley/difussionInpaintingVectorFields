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

def gp_fill(tensor, mask, lengthscale=1.5, variance=1.0, noise=1e-5, use_double=True):
    """
    Fills in missing values (mask == 1) in a tensor using Gaussian Process regression.

    Args:
        tensor: Tensor of shape (1, 2, H, W)
        mask: Binary mask of shape (1, 2, H, W), 0 = known, 1 = unknown
        lengthscale: GP RBF kernel lengthscale
        variance: GP RBF kernel variance
        noise: small base noise added to kernel diagonal
        use_double: whether to cast tensors to float64 for better numerical stability

    Returns:
        Tensor with missing values filled in.
    """
    dtype = torch.float64 if use_double else tensor.dtype
    device = tensor.device

    tensor = tensor.to(dtype)
    mask = mask.to(dtype)

    _, C, H, W = tensor.shape
    filled = tensor.clone()

    for c in range(C):
        channel_data = tensor[0, c]
        channel_mask = mask[0, c]

        known_indices = (channel_mask == 0).nonzero(as_tuple=False)
        unknown_indices = (channel_mask == 1).nonzero(as_tuple=False)

        if known_indices.numel() == 0 or unknown_indices.numel() == 0:
            continue  # nothing to fill

        # Normalize coordinates
        norm_factor = torch.tensor([H, W], dtype=dtype, device=device)
        known_coords = known_indices.float() / norm_factor
        unknown_coords = unknown_indices.float() / norm_factor

        known_values = channel_data[known_indices[:, 0], known_indices[:, 1]]

        # Compute covariance matrices
        K = rbf_kernel(known_coords, known_coords, lengthscale, variance)
        K_s = rbf_kernel(unknown_coords, known_coords, lengthscale, variance)

        # Add noise (jitter) robustly
        jitter = noise
        max_tries = 5
        for _ in range(max_tries):
            try:
                L = torch.linalg.cholesky(K + jitter * torch.eye(K.shape[0], device=device, dtype=dtype))
                break
            except RuntimeError:
                jitter *= 10
        else:
            raise RuntimeError("Cholesky decomposition failed after increasing jitter.")

        K_ss = rbf_kernel(unknown_coords, unknown_coords, lengthscale, variance)
        v = torch.cholesky_solve(known_values.unsqueeze(-1), L)
        pred_mean = K_s @ v

        # Posterior covariance
        cov_post = K_ss - K_s @ torch.cholesky_solve(K_s.T, L)
        cov_post += noise * torch.eye(cov_post.shape[0], device=device, dtype=dtype)  # ensure positive-definite

        # Sample from multivariate normal
        dist = torch.distributions.MultivariateNormal(pred_mean.squeeze(), covariance_matrix=cov_post)
        sampled = dist.sample()

        # Fill with samples instead of mean
        for idx, val in zip(unknown_indices, sampled):
            filled[0, c, idx[0], idx[1]] = val.item()

        # Assign predictions
        for idx, val in zip(unknown_indices, pred_mean):
            filled[0, c, idx[0], idx[1]] = val.item()

    return filled.to(torch.float32)  # Return to float32 for compatibility
