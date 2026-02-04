import torch
import torch.nn as nn
import torch.nn.functional as F
# Assuming this import exists in your environment
from ddpm.helper_functions.compute_divergence import compute_divergence

class PhysicsInformedLoss(nn.Module):
    def __init__(self, weight_fidelity=1.0, weight_physics=1.0, weight_smooth=0.0, 
                 divergence_threshold=0.06):
        """
        Physics-Informed Loss for Vector Field Inpainting.

        Args:
            weight_fidelity (float): Weight for MSE loss against the naive stitch.
            weight_physics (float): Weight for the divergence constraint.
            weight_smooth (float): Weight for smoothness regularization.
            divergence_threshold (float): Only penalize divergence above this threshold.
                                         Default 0.06 is typical for div-free fields.
        """
        super().__init__()
        self.weights = {
            'fidelity': weight_fidelity,
            'physics': weight_physics,
            'smooth': weight_smooth
        }
        self.divergence_threshold = divergence_threshold

        # Register kernel as a buffer so it automatically moves to GPU with the model
        # 3x3 block of ones for 8-neighbor dilation
        self.register_buffer('dilate_kernel', torch.ones((1, 1, 3, 3)))

    @staticmethod
    def _compute_divergence_field(vector_field):
        """
        Computes the divergence field (not mean) for threshold-based penalty.
        """
        # vector_field shape: [B, 2, H, W]
        vx = vector_field[0, 0, :, :]
        vy = vector_field[0, 1, :, :]
        div = compute_divergence(vx, vy)
        return div

    @staticmethod
    def _compute_mean_abs_divergence(vector_field):
        """
        Computes the mean absolute divergence of a vector field.
        """
        # vector_field shape: [B, 2, H, W]
        vx = vector_field[0, 0, :, :]
        vy = vector_field[0, 1, :, :]

        # compute_divergence is imported from helper_functions
        div = compute_divergence(vx, vy)
        return div.abs().mean()

    def _get_boundary_mask(self, mask):
        """
        Creates a boundary mask (seam) where 0 meets 1.
        """
        # Use only first channel for boundary computation
        if mask.shape[1] > 1:
            mask_single = mask[:, 0:1, :, :]
        else:
            mask_single = mask
        mask_float = mask_single.float()

        # Ensure kernel is on same device as mask
        kernel = self.dilate_kernel.to(mask.device)

        # Dilate: padding=1 keeps size identical
        dilated = F.conv2d(mask_float, kernel, padding=1)
        dilated = torch.clamp(dilated, 0, 1)

        # Boundary is where dilation added a 1 that wasn't there before
        boundary = dilated - mask_float
        return boundary

    def forward(self, predicted, known, inpainted, mask):
        """
        Calculates the combined physics loss.

        Args:
            predicted (Tensor): The refined output from the model (Combined/Corrected Field).
            known (Tensor): The ground truth known data.
            inpainted (Tensor): The generated inpainted data (prior to refinement).
            mask (Tensor): Binary mask. Based on recombination logic:
                           1 = Inpainted Region (Hole), 0 = Known Data.

        Returns:
            total_loss (Tensor): The weighted sum of all losses.
            metrics (dict): A dictionary containing individual loss components.
        """

        # --- 1. RECONSTRUCT NAIVE FIELD ---
        # Recreate the stitching logic previously in get_loss
        naive = known * (1 - mask) + (inpainted * mask)

        # --- 2. CALCULATE DYNAMIC THRESHOLDS ---
        # We detach these because they are targets/constraints, not parameters to optimize.
        with torch.no_grad():
            div_known = self._compute_mean_abs_divergence(known)
            div_inp = self._compute_mean_abs_divergence(inpainted)
            # The divergence should not exceed the worst part of the input components
            max_div_threshold = torch.max(div_known, div_inp)

        # --- 3. FIDELITY LOSS ---
        # Only compute fidelity in the INPAINTED region (mask=1)
        # The known region (mask=0) contains true values that shouldn't be modified
        # and the network is constrained to not modify them anyway.
        # Relax fidelity at the boundary where we need freedom to fix divergence
        boundary = self._get_boundary_mask(mask)
        # Expand boundary to match prediction channels if needed
        if boundary.shape[1] == 1 and mask.shape[1] == 2:
            boundary = boundary.expand(-1, 2, -1, -1)
        W_fidelity = mask - (boundary * 0.9)  # Only in inpainted region, less strict at boundary
        W_fidelity = torch.clamp(W_fidelity, 0, 1)  # Ensure non-negative weights

        # Compute fidelity only where W_fidelity > 0 (inpainted region)
        fidelity_diff = W_fidelity * (predicted - naive) ** 2
        # Normalize by the inpainted area to avoid dependence on mask size
        inpainted_area = mask.sum() + 1e-8
        loss_fidelity = fidelity_diff.sum() / inpainted_area

        # --- 4. PHYSICS LOSS (Threshold-based Divergence Penalty) ---
        # Only penalize divergence ABOVE the threshold (normal background level)
        # This prevents the network from just moving divergence around
        div_field = self._compute_divergence_field(predicted)
        abs_div = div_field.abs()
        
        # ReLU-style penalty: only penalize divergence above threshold
        excess_div = F.relu(abs_div - self.divergence_threshold)
        
        # Mean of excess divergence (this is what we want to minimize)
        loss_physics = excess_div.mean()
        
        # Also track the full divergence for metrics
        div_pred = abs_div.mean()

        # --- 5. SMOOTHNESS LOSS (BOUNDARY-ONLY) ---
        # Only apply smoothness at the boundary, not globally
        # This prevents crushing values everywhere while still smoothing the seam
        du = torch.abs(predicted[:, :, :, :-1] - predicted[:, :, :, 1:])
        dv = torch.abs(predicted[:, :, :-1, :] - predicted[:, :, 1:, :])

        # Create boundary weights for smoothness (dilate boundary for wider effect)
        boundary_smooth = self._get_boundary_mask(mask)
        # Dilate boundary mask to cover a few pixels around seam
        kernel = self.dilate_kernel.to(mask.device)
        boundary_smooth = F.conv2d(boundary_smooth.float(), kernel, padding=1)
        boundary_smooth = torch.clamp(boundary_smooth, 0, 1)
        if boundary_smooth.shape[1] == 1:
            boundary_smooth = boundary_smooth.expand(-1, 2, -1, -1)

        # Apply smoothness only at boundary region
        du_weighted = du * boundary_smooth[:, :, :, :-1]
        dv_weighted = dv * boundary_smooth[:, :, :-1, :]
        loss_smooth = torch.mean(du_weighted) + torch.mean(dv_weighted)

        # --- TOTAL LOSS ---
        weighted_fidelity = self.weights['fidelity'] * loss_fidelity
        weighted_physics = self.weights['physics'] * loss_physics
        weighted_smooth = self.weights['smooth'] * loss_smooth

        total_loss = weighted_fidelity + weighted_physics + weighted_smooth

        return total_loss, {
            'loss_fidelity': weighted_fidelity,
            'loss_physics': weighted_physics,
            'loss_smooth': weighted_smooth,
            'metric_div_pred': div_pred,
            'metric_div_thresh': max_div_threshold
        }

# Usage Example:
# criterion = PhysicsInformedLoss()
# loss, stats = criterion(v_refined, v_known, v_generated, mask)
# loss.backward()