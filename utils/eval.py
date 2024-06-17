import torch
from utils.resize_tensor import resize
from utils.image_noiser import generate_noised_tensor_single_step, generate_noised_tensor_iterative
from random import randint


def unnormalize(tensor, mean, std):
    mean = torch.tensor(mean).clone().detach().view(1, -1, 1, 1)
    std = torch.tensor(std).clone().detach().view(1, -1, 1, 1)
    return tensor * std + mean


def evaluate(model, testloader, device):
    model.eval()
    tot_loss = 0
    with torch.no_grad():
        for i, (tensor, label) in enumerate(testloader):
            tensor = tensor.float().to(device)
            mask = (tensor != 0).float().to(device)

            target = generate_noised_tensor_single_step(tensor, target_iteration=randint(1, 1000),
                                                        var_per_iteration=0.005).float().to(device)
            tensor = generate_noised_tensor_iterative(target, iteration=1, variance=0.005).float().to(device)

            tensor = resize(tensor, (3, 64, 128)).to(device)
            target = resize(target, (3, 64, 128)).to(device)
            mask = resize(mask, (3, 64, 128)).to(device)

            output = model(tensor, mask)

            output = output * mask
            target = target * mask
            loss = ((output - target) ** 2 * mask).sum() / mask.sum()
            tot_loss += loss

    avg_loss = tot_loss / len(testloader)
    return avg_loss
