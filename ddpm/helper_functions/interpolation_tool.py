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


def rbf_kernel_legacy(X1, X2, lengthscale=1.0, kernel_noise=1.0):
    """
    Legacy RBF kernel matching reference implementation.
    Uses dist (not squared) with gamma = 1 / lengthscale^2 and kernel_noise^2.
    """
    gamma = 1.0 / (lengthscale ** 2)
    dist = torch.linalg.norm(X1[:, None, :] - X2[None, :, :], dim=2)
    return (kernel_noise ** 2) * torch.exp(-0.5 * gamma * dist)


def incompressible_kernel(X1, X2, lengthscale=1.0, variance=1.0):
    """
    Divergence-free (incompressible) kernel for 2D vector fields.
    Returns a block matrix of shape (2*N1, 2*N2).
    """
    if X1.shape[-1] != 2 or X2.shape[-1] != 2:
        raise ValueError("incompressible_kernel expects 2D coordinates.")

    dx = X1[:, None, 0] - X2[None, :, 0]
    dy = X1[:, None, 1] - X2[None, :, 1]
    r2 = dx.pow(2) + dy.pow(2)

    l2 = lengthscale ** 2
    l4 = l2 ** 2

    base = variance * torch.exp(-0.5 * r2 / l2)
    a = (r2 / l4) - (1.0 / l2)
    b = 1.0 / l4

    k_xx = base * (a - b * dx.pow(2))
    k_xy = base * (-b * dx * dy)
    k_yy = base * (a - b * dy.pow(2))

    n1 = X1.shape[0]
    n2 = X2.shape[0]
    K = torch.zeros((2 * n1, 2 * n2), dtype=base.dtype, device=base.device)

    K[:n1, :n2] = k_xx
    K[:n1, n2:] = k_xy
    K[n1:, :n2] = k_xy
    K[n1:, n2:] = k_yy

    return K


def incompressible_rbf_kernel(X1, X2, lengthscale=1.0, kernel_noise=1.0):
    """
    Incompressible kernel matching reference implementation.
    """
    if X1.shape[-1] != 2 or X2.shape[-1] != 2:
        raise ValueError("incompressible_rbf_kernel expects 2D coordinates.")

    gamma = 1.0 / (lengthscale ** 2)
    dx = X1[:, None, 0] - X2[None, :, 0]
    dy = X1[:, None, 1] - X2[None, :, 1]
    dist = torch.linalg.norm(X1[:, None, :] - X2[None, :, :], dim=2)
    kse = (kernel_noise ** 2) * torch.exp(-0.5 * gamma * dist)

    dxx = -(gamma ** 2) * dx * dx * kse
    dxy = -(gamma ** 2) * dx * dy * kse
    dyy = -(gamma ** 2) * dy * dy * kse

    n1 = X1.shape[0]
    n2 = X2.shape[0]
    K = torch.zeros((2 * n1, 2 * n2), dtype=kse.dtype, device=kse.device)
    K[::2, ::2] = dxx
    K[1::2, ::2] = dxy
    K[::2, 1::2] = dxy
    K[1::2, 1::2] = dyy
    return K

def gp_fill(
    tensor,
    mask,
    lengthscale=1.5,
    variance=1.0,
    noise=1e-5,
    use_double=True,
    max_points=None,
    sample_posterior=False,
    kernel_type="rbf",
    coord_system="pixels",
    return_variance=False,
):
    """
    Fills in missing values (mask == 1) in a tensor using Gaussian Process regression.

    Args:
        tensor: Tensor of shape (1, 2, H, W)
        mask: Binary mask of shape (1, 2, H, W), 0 = known, 1 = unknown
        lengthscale: GP kernel lengthscale (in pixels if coord_system="pixels")
        variance: GP kernel variance
        noise: small base noise added to kernel diagonal
        use_double: whether to cast tensors to float64 for better numerical stability
        max_points: optional cap on number of known points (subsampled if exceeded)
        sample_posterior: if True, sample from GP posterior instead of using mean
        kernel_type: "rbf", "incompressible", "rbf_legacy", "incompressible_rbf"
        coord_system: "pixels" or "normalized" (0-1)
        return_variance: if True, also return per-pixel posterior variance map

    Returns:
        If return_variance is False: Tensor with missing values filled in.
        If return_variance is True: (filled_tensor, variance_map) where
            variance_map is (1, C, H, W) with GP posterior variance at each
            unknown pixel (0 at known pixels).
    """
    orig_device = tensor.device
    if orig_device.type == "mps":
        tensor = tensor.to("cpu")
        mask = mask.to("cpu")
        if use_double:
            use_double = True

    if tensor.device.type == "mps" and use_double:
        use_double = False

    dtype = torch.float64 if use_double else tensor.dtype
    device = tensor.device

    tensor = tensor.to(dtype)
    mask = mask.to(dtype)

    _, C, H, W = tensor.shape
    filled = tensor.clone()
    if return_variance:
        var_map = torch.zeros(1, C, H, W, dtype=dtype, device=device)

    valid_mask_2d = (tensor[0].abs().sum(dim=0) > 1e-5)

    kernel_type = kernel_type.lower()

    coord_system = coord_system.lower()
    use_normalized = coord_system == "normalized"

    if kernel_type in ("incompressible", "incompressible_rbf"):
        if C != 2:
            raise ValueError("incompressible kernel requires 2-channel vector field.")

        channel_mask = mask[0, 0]
        known_indices = ((channel_mask == 0) & valid_mask_2d).nonzero(as_tuple=False)
        unknown_indices = ((channel_mask == 1) & valid_mask_2d).nonzero(as_tuple=False)

        if known_indices.numel() != 0 and unknown_indices.numel() != 0:
            if use_normalized:
                norm_factor = torch.tensor([H, W], dtype=dtype, device=device)
                known_coords = known_indices.to(dtype) / norm_factor
                unknown_coords = unknown_indices.to(dtype) / norm_factor
            else:
                known_coords = known_indices.to(dtype)
                unknown_coords = unknown_indices.to(dtype)

            if max_points is not None and known_indices.shape[0] > max_points:
                perm = torch.randperm(known_indices.shape[0], device=device)
                selected = perm[:max_points]
                known_indices = known_indices[selected]
                known_coords = known_coords[selected]

            known_u = tensor[0, 0][known_indices[:, 0], known_indices[:, 1]]
            known_v = tensor[0, 1][known_indices[:, 0], known_indices[:, 1]]
            known_values = torch.cat([known_u, known_v], dim=0)

            if kernel_type == "incompressible_rbf":
                K = incompressible_rbf_kernel(known_coords, known_coords, lengthscale, variance)
                K_s = incompressible_rbf_kernel(unknown_coords, known_coords, lengthscale, variance)
            else:
                K = incompressible_kernel(known_coords, known_coords, lengthscale, variance)
                K_s = incompressible_kernel(unknown_coords, known_coords, lengthscale, variance)

            jitter = max(noise, 1e-6)
            max_tries = 10
            for _ in range(max_tries):
                try:
                    K_sym = 0.5 * (K + K.T)
                    L = torch.linalg.cholesky(K_sym + jitter * torch.eye(K.shape[0], device=device, dtype=dtype))
                    break
                except RuntimeError:
                    jitter *= 10
            else:
                raise RuntimeError("Cholesky decomposition failed after increasing jitter.")

            v = torch.cholesky_solve(known_values.unsqueeze(-1), L)
            pred_mean = K_s @ v

            if sample_posterior:
                if kernel_type == "incompressible_rbf":
                    K_ss = incompressible_rbf_kernel(unknown_coords, unknown_coords, lengthscale, variance)
                else:
                    K_ss = incompressible_kernel(unknown_coords, unknown_coords, lengthscale, variance)
                cov_post = K_ss - K_s @ torch.cholesky_solve(K_s.T, L)
                cov_post += noise * torch.eye(cov_post.shape[0], device=device, dtype=dtype)
                dist = torch.distributions.MultivariateNormal(pred_mean.squeeze(), covariance_matrix=cov_post)
                pred_values = dist.sample()
            else:
                pred_values = pred_mean.squeeze(-1)

            if return_variance:
                # Posterior variance for incompressible kernel.
                # Diagonal of K_ss - K_s @ K^{-1} @ K_s^T
                if kernel_type == "incompressible_rbf":
                    K_ss = incompressible_rbf_kernel(unknown_coords, unknown_coords, lengthscale, variance)
                else:
                    K_ss = incompressible_kernel(unknown_coords, unknown_coords, lengthscale, variance)
                alpha_var = torch.cholesky_solve(K_s.T, L)
                var_reduction = (K_s * alpha_var.T).sum(dim=1)
                K_ss_diag = K_ss.diag()
                post_var = (K_ss_diag - var_reduction).clamp(min=0)
                n_unk = unknown_indices.shape[0]
                for idx_i, idx in enumerate(unknown_indices):
                    var_map[0, 0, idx[0], idx[1]] = post_var[idx_i].item()
                    var_map[0, 1, idx[0], idx[1]] = post_var[n_unk + idx_i].item()

            if pred_values.ndim == 0:
                pred_values = pred_values.unsqueeze(0)

            n_unknown = unknown_indices.shape[0]
            pred_u = pred_values[:n_unknown]
            pred_v = pred_values[n_unknown:]

            for idx, val in zip(unknown_indices, pred_u):
                filled[0, 0, idx[0], idx[1]] = val.item()
            for idx, val in zip(unknown_indices, pred_v):
                filled[0, 1, idx[0], idx[1]] = val.item()
    else:
        for c in range(C):
            channel_data = tensor[0, c]
            channel_mask = mask[0, c]

            known_indices = ((channel_mask == 0) & valid_mask_2d).nonzero(as_tuple=False)
            unknown_indices = ((channel_mask == 1) & valid_mask_2d).nonzero(as_tuple=False)

            if known_indices.numel() == 0 or unknown_indices.numel() == 0:
                continue  # nothing to fill

            if use_normalized:
                norm_factor = torch.tensor([H, W], dtype=dtype, device=device)
                known_coords = known_indices.to(dtype) / norm_factor
                unknown_coords = unknown_indices.to(dtype) / norm_factor
            else:
                known_coords = known_indices.to(dtype)
                unknown_coords = unknown_indices.to(dtype)

            if max_points is not None and known_indices.shape[0] > max_points:
                perm = torch.randperm(known_indices.shape[0], device=device)
                selected = perm[:max_points]
                known_indices = known_indices[selected]
                known_coords = known_coords[selected]

            known_values = channel_data[known_indices[:, 0], known_indices[:, 1]]

            # Compute covariance matrices
            if kernel_type == "rbf_legacy":
                K = rbf_kernel_legacy(known_coords, known_coords, lengthscale, variance)
                K_s = rbf_kernel_legacy(unknown_coords, known_coords, lengthscale, variance)
            else:
                K = rbf_kernel(known_coords, known_coords, lengthscale, variance)
                K_s = rbf_kernel(unknown_coords, known_coords, lengthscale, variance)

            # Add noise (jitter) robustly
            jitter = max(noise, 1e-6)
            max_tries = 10
            for _ in range(max_tries):
                try:
                    K_sym = 0.5 * (K + K.T)
                    L = torch.linalg.cholesky(K_sym + jitter * torch.eye(K.shape[0], device=device, dtype=dtype))
                    break
                except RuntimeError:
                    jitter *= 10
            else:
                raise RuntimeError("Cholesky decomposition failed after increasing jitter.")

            v = torch.cholesky_solve(known_values.unsqueeze(-1), L)
            pred_mean = K_s @ v

            if sample_posterior:
                if kernel_type == "rbf_legacy":
                    K_ss = rbf_kernel_legacy(unknown_coords, unknown_coords, lengthscale, variance)
                else:
                    K_ss = rbf_kernel(unknown_coords, unknown_coords, lengthscale, variance)
                cov_post = K_ss - K_s @ torch.cholesky_solve(K_s.T, L)
                cov_post += noise * torch.eye(cov_post.shape[0], device=device, dtype=dtype)
                dist = torch.distributions.MultivariateNormal(pred_mean.squeeze(), covariance_matrix=cov_post)
                pred_values = dist.sample()
            else:
                pred_values = pred_mean.squeeze(-1)

            if return_variance:
                # Efficient posterior variance: diag(K_ss) - diag(K_s @ K^{-1} @ K_s^T)
                # diag(K_ss) = prior variance at each point (kernel self-covariance)
                if kernel_type == "rbf_legacy":
                    prior_var = variance ** 2  # rbf_legacy uses kernel_noise^2
                else:
                    prior_var = variance  # standard rbf
                alpha_var = torch.cholesky_solve(K_s.T, L)  # (n_known, n_unknown)
                var_reduction = (K_s * alpha_var.T).sum(dim=1)  # (n_unknown,)
                post_var = (prior_var - var_reduction).clamp(min=0)
                for idx_i, idx in enumerate(unknown_indices):
                    var_map[0, c, idx[0], idx[1]] = post_var[idx_i].item()

            if pred_values.ndim == 0:
                pred_values = pred_values.unsqueeze(0)

            for idx, val in zip(unknown_indices, pred_values):
                filled[0, c, idx[0], idx[1]] = val.item()

    if return_variance:
        return filled.to(torch.float32).to(orig_device), var_map.to(torch.float32).to(orig_device)
    return filled.to(torch.float32).to(orig_device)  # Return to original device
