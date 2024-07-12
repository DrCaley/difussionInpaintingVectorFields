import torch
from utils.stream_flow import calculate_flow


def flow_MSE(predicted, target):
    flow_diff = torch.abs(calculate_flow(predicted)) - torch.abs(calculate_flow(target))
    flow_diff[flow_diff < 0.0] = 0.0
    flow_MSE = (flow_diff ** 2).sum() / torch.ones_like(flow_diff).sum()

    return flow_MSE


def MSE_with_flow(predicted, target, mask, weight=1.0):
    predicted = predicted * mask
    target = target * mask
    flow_diff = torch.abs(calculate_flow(predicted)) - torch.abs(calculate_flow(target))
    flow_diff = flow_diff * mask
    flow_diff[flow_diff < 0.0] = 0.0

    current_MSE = ((predicted - target) ** 2).sum() / mask.sum()
    flow_MSE = (flow_diff ** 2).sum() / mask.sum()

    if(flow_MSE > 0):
        print(f"Flow_MSE: {flow_MSE}")

    return current_MSE + flow_MSE * weight
