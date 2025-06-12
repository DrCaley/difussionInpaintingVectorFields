import torch
import torch.nn.functional as F


def interpolate_masked_velocity_field(velocity: torch.Tensor, mask: torch.Tensor,
                                              num_iters: int = 50) -> torch.Tensor:
    """
    PyTorch-only interpolation of a masked 2-channel velocity field using convolutional smoothing.

    Args:
        velocity (torch.Tensor): Tensor of shape (2, H, W). Known values are in mask == 0 region.
        mask (torch.Tensor): Tensor of shape (1, H, W), where 1 means missing (to be interpolated),
                             and 0 means known.
        num_iters (int): Number of smoothing iterations to propagate known values.

    Returns:
        torch.Tensor: Filled velocity field of shape (2, H, W).
    """
    assert velocity.shape[0] == 2, "Velocity must have shape (2, H, W)"
    assert mask.shape[0] == 1, "Mask must have shape (1, H, W)"

    # Combine mask with velocity: 1 where needs filling, 0 where known
    known_mask = 1 - mask  # 1 where known
    filled = velocity.clone()

    # 3x3 averaging kernel
    kernel = torch.ones(1, 1, 3, 3, device=velocity.device) / 9.0
    padding = 1

    for _ in range(num_iters):
        for c in range(2):  # u and v channels
            v = filled[c:c + 1]
            v_known = v * known_mask

            # Convolve known values and mask separately
            smoothed_vals = F.conv2d(v_known.unsqueeze(0), kernel, padding=padding)
            smoothed_mask = F.conv2d(known_mask.unsqueeze(0), kernel, padding=padding)

            # Avoid divide-by-zero
            smoothed_mask = smoothed_mask + 1e-6

            # Compute new estimates only where mask == 1 (unknown)
            interpolated = smoothed_vals / smoothed_mask
            filled[c] = interpolated.squeeze(0) * mask + v.squeeze(0) * known_mask

    return filled
