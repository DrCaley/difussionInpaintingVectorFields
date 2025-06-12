import torch
import torch.nn.functional as F

def interpolate_masked_velocity_field(velocity: torch.Tensor, mask: torch.Tensor, max_iters: int = 500) -> torch.Tensor:
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
