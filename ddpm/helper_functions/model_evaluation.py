import torch
from torch import nn

from data_prep.data_initializer import DDInitializer

def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    count = 0

    criterion = nn.MSELoss()

    with torch.no_grad():
        for x0, t, epsilon in data_loader:
            x0 = x0.to(device)
            t = t.to(device)
            epsilon = epsilon.to(device)
            n = len(x0)

            noisy_imgs = model(x0, t, epsilon)
            epsilon_theta = model.backward(noisy_imgs, t.reshape(n, -1))
            loss = criterion(epsilon_theta, epsilon)
            total_loss += loss.item() * n
            count += n

    return total_loss / count