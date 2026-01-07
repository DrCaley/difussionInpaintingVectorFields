import torch
import torch.nn as nn
import torch.nn.functional as F
# Assuming this import exists in your environment
from ddpm.helper_functions.compute_divergence import compute_divergence

class PhysicsInformedLoss(nn.Module):
    def __init__(self, weight_fidelity=1.0, weight_physics=1.0, weight_smooth=0.1, weight_known=10.0):
        """
        Physics-Informed Loss for Vector Field Inpainting.
        
        NEW APPROACH: Directly minimize divergence at boundary while preserving
        values away from boundary.

        Args:
            weight_fidelity (float): Weight for preserving values AWAY from boundary.
            weight_physics (float): Weight for minimizing divergence AT boundary.
            weight_smooth (float): Weight for smoothness regularization.
            weight_known (float): Weight for keeping known region UNCHANGED.
        """
        super().__init__()
        self.weights = {
            'fidelity': weight_fidelity,
            'physics': weight_physics,
            'smooth': weight_smooth,
            'known': weight_known
        }

        # Register kernel as a buffer so it automatically moves to GPU with the model
        # 3x3 block of ones for 8-neighbor dilation
        self.register_buffer('dilate_kernel', torch.ones((1, 1, 3, 3)))

    def _compute_divergence_map(self, vector_field):
        """
        Computes the divergence at each pixel of a vector field.
        Returns a [H-1, W-1] map of divergence values.
        """
        # vector_field shape: [B, 2, H, W]
        vx = vector_field[0, 0, :, :]
        vy = vector_field[0, 1, :, :]

        # compute_divergence returns a divergence map
        div = compute_divergence(vx, vy)
        return div

    def _get_boundary_mask(self, mask, width=2):
        """
        Creates a boundary mask (seam) where 0 meets 1.
        Handles masks with shape [B, C, H, W] where C can be 1 or 2.
        
        Args:
            mask: Binary mask [B, C, H, W]
            width: How many pixels wide the boundary region should be
        """
        mask_float = mask.float()
        
        # If mask has 2 channels (broadcasted), just use one channel for boundary detection
        if mask_float.shape[1] == 2:
            mask_float = mask_float[:, 0:1, :, :]  # Use first channel only [B, 1, H, W]

        # Multiple dilations/erosions for wider boundary
        dilated = mask_float.clone()
        eroded = mask_float.clone()
        
        for _ in range(width):
            dilated = F.conv2d(dilated, self.dilate_kernel, padding=1)
            dilated = torch.clamp(dilated, 0, 1)
            
            eroded = F.conv2d(1 - eroded, self.dilate_kernel, padding=1)
            eroded = 1 - torch.clamp(eroded, 0, 1)

        # Boundary is the transition zone
        boundary = dilated - eroded
        boundary = torch.clamp(boundary, 0, 1)
        
        # If original mask had 2 channels, broadcast boundary back to 2 channels
        if mask.shape[1] == 2:
            boundary = boundary.expand(-1, 2, -1, -1)
        
        return boundary

    def forward(self, predicted, known, inpainted, mask):
        """
        Calculates the combined physics loss.
        
        LOSS DESIGN:
        1. KNOWN FIDELITY: Don't change the known region at all
        2. MINIMAL CHANGE: Minimize total changes to the field
        3. BOUNDARY DIVERGENCE: Minimize |div| at boundary pixels
        4. SMOOTHNESS: Penalize sharp gradients at boundary

        Args:
            predicted (Tensor): The refined output from the model.
            known (Tensor): The known data (outside mask).
            inpainted (Tensor): The inpainted data (inside mask).
            mask (Tensor): Binary mask. 1 = Inpainted Region, 0 = Known Data.

        Returns:
            total_loss (Tensor): The weighted sum of all losses.
            metrics (dict): A dictionary containing individual loss components.
        """

        # --- 1. RECONSTRUCT NAIVE FIELD ---
        naive = known * (1 - mask) + (inpainted * mask)

        # --- 2. GET BOUNDARY MASK ---
        boundary = self._get_boundary_mask(mask, width=2)
        
        # Known region mask (where mask = 0)
        known_region = (1 - mask)

        # --- 3. KNOWN REGION LOSS (keep known data UNCHANGED) ---
        # This is critical: the known region is noised ground truth, don't modify it
        loss_known = torch.mean(known_region * (predicted - known) ** 2)

        # --- 4. MINIMAL CHANGE LOSS (minimize total changes to the field) ---
        # Penalize any deviation from the naive stitch
        loss_change = torch.mean((predicted - naive) ** 2)

        # --- 5. BOUNDARY DIVERGENCE LOSS (minimize divergence AT boundary) ---
        div_map = self._compute_divergence_map(predicted)
        
        # Get boundary mask at divergence resolution
        boundary_single = boundary[:, 0:1, :, :]  # [B, 1, H, W]
        boundary_div = boundary_single.squeeze()  # [H, W]
        
        # Ensure shapes match
        H_div, W_div = div_map.shape
        boundary_div = boundary_div[:H_div, :W_div]
        
        # Penalize divergence magnitude at boundary
        boundary_div_values = torch.abs(div_map) * boundary_div
        loss_boundary_div = boundary_div_values.sum() / (boundary_div.sum() + 1e-8)

        # --- 6. SMOOTHNESS LOSS (encourage smooth transition at boundary) ---
        du = torch.abs(predicted[:, :, :, :-1] - predicted[:, :, :, 1:])
        dv = torch.abs(predicted[:, :, :-1, :] - predicted[:, :, 1:, :])
        
        boundary_h = boundary[:, :, :, :-1]
        boundary_v = boundary[:, :, :-1, :]
        
        loss_smooth = torch.mean(boundary_h * du) + torch.mean(boundary_v * dv)

        # --- TOTAL LOSS ---
        weighted_known = self.weights['known'] * loss_known
        weighted_change = self.weights['fidelity'] * loss_change
        weighted_boundary_div = self.weights['physics'] * loss_boundary_div
        weighted_smooth = self.weights['smooth'] * loss_smooth

        total_loss = weighted_known + weighted_change + weighted_boundary_div + weighted_smooth

        return total_loss, {
            'loss_known': weighted_known,
            'loss_change': weighted_change,
            'loss_boundary_div': weighted_boundary_div,
            'loss_smooth': weighted_smooth,
            'metric_boundary_div': loss_boundary_div.detach()
        }


# Usage Example:
# criterion = PhysicsInformedLoss()
# loss, stats = criterion(v_refined, v_known, v_generated, mask)
# loss.backward()