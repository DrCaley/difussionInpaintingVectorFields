import torch
from utils.stream_flow import calculate_flow


def flow_mse(predicted, target, mask=None, weight=1.0):
    """MSE where error is how much predicted pixels are worse at following the incompressible stream function than the target"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predicted = predicted.to(device)
    target = target.to(device)

    if mask is None:
        mask = torch.ones_like(target).to(device)

    mask = mask.to(device)
    predicted = predicted * mask
    target = target * mask

    flow_diff = torch.abs(calculate_flow(predicted)) - torch.abs(calculate_flow(target))
    flow_diff = flow_diff * mask
    flow_diff[flow_diff < 0.0] = 0.0

    mse = (flow_diff ** 2).sum() / mask.sum()

    return mse * weight
