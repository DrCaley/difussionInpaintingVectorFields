import torch
from torch import nn

from data_prep.data_initializer import DDInitializer

data_init = DDInitializer()
noise_strat = data_init.noise_strategy

def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    count = 0

    criterion = nn.MSELoss()

    with torch.no_grad():
        for batch in data_loader:
            x0 = batch[0].to(device).float()
            n = len(x0)

            t = torch.randint(0, model.n_steps, (n,)).to(device)
            epsilon = noise_strat(x0, t)

            noisy_imgs = model(x0, t, epsilon)
            epsilon_theta = model.backward(noisy_imgs, t.reshape(n, -1))
            loss = criterion(epsilon_theta, epsilon)
            total_loss += loss.item() * n
            count += n

    return total_loss / count