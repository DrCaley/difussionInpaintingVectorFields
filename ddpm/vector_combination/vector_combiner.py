import torch
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path

from ddpm.helper_functions.HH_decomp import decompose_vector_field
from ddpm.vector_combination.combiner_unet import VectorCombinationUNet
from ddpm.vector_combination.combination_loss import PhysicsInformedLoss
from data_prep.data_initializer import DDInitializer


# Global cache for pretrained model (loaded once, reused)
_pretrained_combnet = None
_pretrained_combnet_path = None


def get_pretrained_combnet(device, dd=None):
    """
    Load pretrained CombNet (cached for reuse).
    
    Returns None if no pretrained model exists or if disabled in config.
    """
    global _pretrained_combnet, _pretrained_combnet_path
    
    # Get path from config or use default
    if dd is not None:
        pretrained_path = dd.get_attribute("pretrained_combnet_path")
        if pretrained_path is None or pretrained_path == "null":
            return None  # Explicitly disabled
    else:
        pretrained_path = None
    
    if pretrained_path is None:
        # Default path
        pretrained_path = Path(__file__).parent.parent / "Trained_Models" / "pretrained_combnet.pt"
    else:
        pretrained_path = Path(pretrained_path)
        # Handle relative paths
        if not pretrained_path.is_absolute():
            pretrained_path = Path(__file__).parent.parent.parent / pretrained_path
    
    # Return cached model if same path
    if _pretrained_combnet is not None and _pretrained_combnet_path == str(pretrained_path):
        return _pretrained_combnet
    
    if not pretrained_path.exists():
        return None
    
    try:
        checkpoint = torch.load(pretrained_path, map_location=device, weights_only=False)
        config = checkpoint.get('config', {'n_channels': 4, 'n_classes': 2})
        
        model = VectorCombinationUNet(
            n_channels=config.get('n_channels', 4),
            n_classes=config.get('n_classes', 2)
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        _pretrained_combnet = model
        _pretrained_combnet_path = str(pretrained_path)
        print(f"Loaded pretrained CombNet from {pretrained_path}")
        return model
    except Exception as e:
        print(f"Warning: Could not load pretrained CombNet: {e}")
        return None


def combine_fields(known, inpainted, mask):
    """
    Combine known and inpainted regions.
    
    If use_comb_net is enabled:
      - First tries to use pretrained CombNet (fast, single forward pass)
      - Falls back to per-step training if no pretrained model exists
    Otherwise, use naive stitching.
    """
    dd = DDInitializer()
    naive = known * (1 - mask) + (inpainted * mask)
    
    if dd.get_use_comb_net():
        device = naive.device
        
        # Try pretrained model first (FAST path)
        pretrained = get_pretrained_combnet(device, dd)
        if pretrained is not None:
            return apply_pretrained_combnet(pretrained, naive, mask)
        
        # Fall back to per-step training (SLOW path)
        print("Warning: No pretrained CombNet found, using slow per-step training")
        with torch.enable_grad():
            return train_boundary_fixer(dd, naive, known, inpainted, mask)
    else:
        return naive


def apply_pretrained_combnet(model, naive, mask):
    """
    Apply pretrained CombNet in a single forward pass (no training).
    
    This is ~200x faster than train_boundary_fixer since we skip training.
    """
    # Combine inputs: [naive (2ch), mask (2ch)] = 4 channels
    combined_input = torch.cat([naive, mask], dim=1)
    
    with torch.no_grad():
        prediction = model(combined_input)
    
    return prediction


def train_boundary_fixer(dd, naive, known, inpainted, mask):
    """
    Train a small network to fix boundary divergence.
    
    New approach: Loss directly minimizes divergence at boundary while
    preserving values away from boundary.
    """
    device = naive.device
    
    # Combine inputs: [naive (2ch), mask (2ch)] = 4 channels
    combined_input = torch.cat([naive, mask], dim=1)
    
    # Small UNet for boundary correction
    unet = VectorCombinationUNet(n_channels=4, n_classes=2).to(device)
    unet.train()
    
    # Higher learning rate for faster convergence
    optimizer = optim.Adam(unet.parameters(), lr=1e-2)
    
    # New loss that directly targets boundary divergence
    loss_fn = PhysicsInformedLoss(
        weight_fidelity=dd.get_attribute("fidelity_weight"),  # Preserve away from boundary
        weight_physics=dd.get_attribute("physics_weight"),     # Minimize boundary divergence
        weight_smooth=dd.get_attribute("smooth_weight")        # Smooth transition
    ).to(device)
    
    num_steps = dd.get_attribute("comb_training_steps")
    
    for i in range(num_steps):
        optimizer.zero_grad()
        prediction = unet(combined_input)
        loss, stats = loss_fn(prediction, known, inpainted, mask)
        loss.backward()
        optimizer.step()
    
    return prediction.detach()


def apply_final_boundary_fix(result, mask):
    """
    Apply local smoothing to reduce boundary discontinuities.
    
    Uses iterative local averaging only at the narrow boundary region.
    This preserves values away from the boundary while smoothing the transition.
    
    Args:
        result: Tensor [B, 2, H, W] - the inpainted result
        mask: Tensor [B, 2, H, W] - binary mask (1 = inpainted region)
        
    Returns:
        Tensor [B, 2, H, W] - result with smoother boundaries
    """
    device = result.device
    B, C, H, W = result.shape
    
    final = result.clone()
    
    for b in range(B):
        # Get narrow boundary mask (2-3 pixels on each side)
        mask_single = mask[b, 0:1, :, :].float()  # [1, H, W]
        
        # Dilate
        kernel = torch.ones((1, 1, 3, 3), device=device)
        dilated = F.conv2d(mask_single.unsqueeze(0), kernel, padding=1)
        dilated = torch.clamp(dilated, 0, 1).squeeze(0)
        
        # Erode
        eroded = F.conv2d((1-mask_single).unsqueeze(0), kernel, padding=1)
        eroded = 1 - torch.clamp(eroded, 0, 1).squeeze(0)
        
        # Narrow boundary (just 1-2 pixels on each side)
        boundary = (dilated - mask_single) + (mask_single - eroded)
        boundary = boundary.expand(2, -1, -1)  # [2, H, W]
        
        # Apply local smoothing only at boundary
        field = final[b].clone()
        avg_kernel = torch.ones((2, 1, 3, 3), device=device) / 9.0
        
        # Multiple passes of gentle smoothing
        for _ in range(5):
            local_avg = F.conv2d(field.unsqueeze(0), avg_kernel, padding=1, groups=2).squeeze(0)
            # Blend: 50% smoothing at boundary only
            field = field * (1 - 0.5 * boundary) + local_avg * (0.5 * boundary)
        
        final[b] = field
    
    return final
