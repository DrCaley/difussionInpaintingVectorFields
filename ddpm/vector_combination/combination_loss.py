import torch
import torch.nn as nn
import torch.nn.functional as F
# Assuming this import exists in your environment
from ddpm.helper_functions.compute_divergence import compute_divergence

class PhysicsInformedLoss(nn.Module):
    def __init__(self, weight_fidelity=1.0, weight_physics=1.0):
        """
        Physics-Informed Loss for Vector Field Inpainting.

        Args:
            weight_fidelity (float): Weight for MSE loss against the naive stitch.
            weight_physics (float): Weight for the divergence constraint.
        """
        super().__init__()
        self.weights = {
            'fidelity': weight_fidelity,
            'physics': weight_physics,
        }

        # Register kernel as a buffer so it automatically moves to GPU with the model
        # 3x3 block of ones for 8-neighbor dilation
        self.register_buffer('dilate_kernel', torch.ones((1, 1, 3, 3)))

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
        # We define a weight map to relax constraints at the stitching seam
        # boundary = self._get_boundary_mask(mask)
        # W_fidelity = torch.ones_like(mask) - (boundary * 0.9)

        # Current implementation: Trust data equally everywhere
        fidelity_mask = torch.ones_like(mask)

        loss_fidelity = torch.mean(fidelity_mask * (predicted - naive)) ** 2

        # --- 4. PHYSICS LOSS (Divergence Constraint) ---
        div_pred = self._compute_mean_abs_divergence(predicted)

        # Penalty: ReLU(|Div_Pred| - |Div_Threshold|)
        # Only penalize if divergence is WORSE than the inputs
        loss_physics = F.relu(div_pred - max_div_threshold)


        # # --- MAGNITUDE LOSS ---
        # pred_mag = torch.mean(torch.abs(predicted))
        # known_mag = torch.mean(torch.abs(known))
        # loss_mag = torch.abs(pred_mag - known_mag)

        # --- TOTAL LOSS ---
        weighted_fidelity = self.weights['fidelity'] * loss_fidelity
        weighted_physics = self.weights['physics'] * loss_physics

        total_loss = weighted_fidelity + weighted_physics #+ loss_mag

        return total_loss, {
            'total_loss': total_loss,
            'loss_fidelity': weighted_fidelity,
            'loss_physics': weighted_physics,
            'metric_div_pred': div_pred,
            'metric_div_thresh': max_div_threshold
        }

# Usage Example:
# criterion = PhysicsInformedLoss()
# loss, stats = criterion(v_refined, v_known, v_generated, mask)
# loss.backward()