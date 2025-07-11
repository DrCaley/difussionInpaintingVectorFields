import torch
from torch import nn

from data_prep.data_initializer import DDInitializer

def evaluate(model, data_loader, device):
    dd = DDInitializer()

    model.eval()
    total_loss = 0.0
    count = 0

    criterion = nn.MSELoss()

    with torch.no_grad():
        for i, (x0, t, epsilon) in enumerate(data_loader):
            if i > 20 : #perhaps we can evaluate a small sample instead of all data?
                break
            x0 = x0.to(device)
            t = t.to(device)
            epsilon = epsilon.to(device)
            n = len(x0)

            x0_reshaped = torch.permute(x0, (1, 2, 3, 0)).to(device)
            mask_raw = (dd.get_standardizer().unstandardize(x0_reshaped).abs() != 0.0).float().to(device)
            mask = torch.permute(mask_raw, (3, 0, 1, 2)).to(device)

            noisy_imgs = model(x0, t, epsilon)
            epsilon_theta, _ = model.backward(noisy_imgs, t.reshape(n, -1), mask)
            loss = criterion(epsilon_theta, epsilon)
            total_loss += loss.item() * n
            count += n

    return total_loss / count