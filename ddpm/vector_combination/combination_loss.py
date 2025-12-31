import torch
import torch.nn as nn
import torch.nn.functional as F
from ddpm.helper_functions.compute_divergence import compute_divergence
import csv

class PhysicsLoss(nn.Module):
    def __init__(self, weight_fidelity=1.0, weight_physics=1.0, weight_smooth=0.1):
        super().__init__()
        self.w_fid = weight_fidelity
        self.w_phys = weight_physics
        self.w_smooth = weight_smooth

        # Kernel for "dilating" the mask to find boundaries
        # A 3x3 block of ones effectively looks at all neighbors
        self.dilate_kernel = torch.ones((1, 1, 3, 3))

    def get_boundary_mask(self, binary_mask):
        """
        Creates a 'Seam Mask' by finding pixels where 0 meets 1.
        """
        # Ensure mask is float for conv2d
        mask_float = binary_mask.float()

        # Dilate: If any neighbor is 1, output becomes 1
        # padding=1 ensures output size stays same
        dilated = F.conv2d(mask_float, self.dilate_kernel.to(mask_float.device), padding=1)
        dilated = torch.clamp(dilated, 0, 1) # Clamp to binary

        # Erode: (Optional, or just subtract original from dilated)
        # Boundary = Dilated - Original (Points that were 0 but are next to 1)
        # Note: Depending on if you want the boundary on the 'known' or 'unknown' side,
        # you can adjust this logic. This captures the 'unknown' side of the seam.
        boundary = dilated - mask_float

        return boundary

    def forward(self, v_combined, v_naive, max_div, mask):
        """
        v_pred:  The output from the U-Net (Corrected Field)
        v_naive: The original stitched input
        mask:    Binary mask (1 = Known Data, 0 = Hole)
        """

        # --- 1. PREPARE MAPS ---
        # Find the seam where the stitching happened
        #boundary = self.get_boundary_mask(mask)

        # Create Fidelity Weight Map (W)
        # Trust data everywhere (1.0) EXCEPT at boundary (0.1)
        # We give the model freedom to change the seam.
        W_fidelity = torch.ones_like(mask)
        W_fidelity = W_fidelity #- (boundary * 0.9) # 1.0 everywhere, 0.1 at seam

        # --- 2. FIDELITY LOSS (Data Consistency) ---
        # Weighted MSE: Penalize changes, but penalize them LESS at the seam
        diff = (v_combined - v_naive) ** 2
        loss_fidelity = torch.mean(W_fidelity * diff)

        # --- 3. PHYSICS LOSS (Divergence Inequality) ---
        divergence = get_mean_abs_div(v_combined)

        # Constraint: Don't make divergence WORSE than the original.
        # ReLU( |Div_Pred| - |Div_Thresh| )
        # If Pred is lower, loss is 0. If Pred is higher, loss is positive.
        divergence_penalty = F.relu(divergence - max_div)

        # Option: Weight divergence higher at the boundary?
        # For now, we apply it globally, but you could multiply by 'boundary' here.
        loss_physics = torch.mean(divergence_penalty) # fixme why is this zero for the naive result?


        # --- 4. SMOOTHNESS LOSS (Regularization) ---
        # Since we lowered fidelity at the seam, we need to enforce smoothness there
        # to prevent jagged artifacts.
        # Calculate gradients of velocity
        du = torch.abs(v_combined[:, :, :, :-1] - v_combined[:, :, :, 1:]) # Horizontal changes
        dv = torch.abs(v_combined[:, :, :-1, :] - v_combined[:, :, 1:, :]) # Vertical changes

        # Only penalize "jerky" movements at the boundary
        # We need to align the mask to the gradient sizes (they lose 1 pixel)
        # This is a simplified version; normally we crop the mask.
        loss_smooth = torch.mean(du) + torch.mean(dv)


        # --- TOTAL LOSS ---
        return (self.w_fid * loss_fidelity), (self.w_phys * loss_physics), (self.w_smooth * loss_smooth)

def get_loss(v_combined, v_known, v_inpainted, mask):
    loss = PhysicsLoss()
    known_div = get_mean_abs_div(v_known)
    inpainted_div = get_mean_abs_div(v_inpainted)
    div_thresh = max(known_div, inpainted_div)
    v_naive = v_known * (1 - mask) + (v_inpainted * mask) #This step is the one where the recombination happens
    return loss.forward(v_combined, v_naive, div_thresh, mask)


def get_mean_abs_div(vector_field):
    vx = vector_field[0,0,:,:]
    vy = vector_field[0,1,:,:]
    div = compute_divergence(vx, vy)
    abs_div = div.abs()
    mean_abs_div = abs_div.mean()
    return mean_abs_div
