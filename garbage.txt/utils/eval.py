import torch
from utils.resize_tensor import resize
from utils.image_noiser import generate_noised_tensor_single_step, generate_noised_tensor_iterative
from random import randint
#from utils.tensors_to_png import generate_png


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
            img_size = torch.ones((5, 2, 28, 28))
            mask = resize((img_size != 0).float(), (2, 64, 128)).to(device)

            target = generate_noised_tensor_single_step(tensor, target_iteration=randint(1, 1000),
                                                        var_per_iteration=0.005).float().to(device)
            tensor = generate_noised_tensor_iterative(target, iteration=1, variance=0.005).float().to(device)

            tensor = resize(tensor, (2, 64, 128)).to(device)
            target = resize(target, (2, 64, 128)).to(device)

            output = model(tensor, mask)

            #generate_png(torch.concat((output[0], mask[0][0:1]), dim=1))

            output = output * mask
            target = target * mask
            loss = ((output - target) ** 2 * mask).sum() / mask.sum()
            tot_loss += loss

    avg_loss = tot_loss / len(testloader)
    return avg_loss
