import math
import random
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, Lambda

from dataloaders.dataloader import OceanImageDataset
from medium_ddpm.dir.ddpm import MyDDPM
from medium_ddpm.dir.resize_tensor import resize
from medium_ddpm.dir.unet_resized import MyUNet
from medium_ddpm.dir.utils import show_images, generate_new_images
from utils import tensors_to_png

# Setting reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Definitions
STORE_PATH_FASHION = "ddpm_model_fashion.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_steps, min_beta, max_beta = 1000, 0.0001, 0.02
ddpm = MyDDPM(MyUNet(n_steps), n_steps=n_steps, min_beta=min_beta, max_beta=max_beta, device=device)

no_train = False
batch_size = 1
n_epochs = 1
lr = 0.001


class ResizeTransform:
    def __init__(self, end_shape):
        self.end_shape = end_shape

    def __call__(self, tensor):
        return resize(tensor, self.end_shape).float()

# Initialize dataset with transformations
transform = Compose([
    Lambda(lambda x: (x - 0.5) * 2),     # Normalize to range [-1, 1]
    ResizeTransform((2, 64, 128))        # Resized to (1, 64, 128)
])

store_path = "../../../models/ddpm_ocean_v0.pt"

data = OceanImageDataset(
    mat_file="../../../data/rams_head/stjohn_hourly_5m_velocity_ramhead_v2.mat",
    boundaries="../../../data/rams_head/boundaries.yaml",
    num=10,
    transform=transform
)

train_len = int(math.floor(len(data) * 0.7))
test_len = int(math.floor(len(data) * 0.15))
val_len = len(data) - train_len - test_len

training_data, test_data, validation_data = random_split(data, [train_len, test_len, val_len])

loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)
val_loader = DataLoader(validation_data, batch_size=batch_size)


class CustomLoss(nn.Module):
    def __init__(self, non_zero_weight=5.0, zero_weight=1.0):
        super(CustomLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.mae = nn.L1Loss(reduction='none')
        self.non_zero_weight = non_zero_weight
        self.zero_weight = zero_weight

    def forward(self, eta_theta, eta, x0):
        non_zero_mask = (x0 != 0).float()
        zero_mask = (x0 == 0).float()

        mse_loss = self.mse(eta_theta, eta)
        mae_loss = self.mae(eta_theta, eta)

        weighted_mse_loss = mse_loss * (self.non_zero_weight * non_zero_mask + self.zero_weight * zero_mask)
        weighted_mae_loss = mae_loss * (self.non_zero_weight * non_zero_mask + self.zero_weight * zero_mask)

        weighted_mse_loss = weighted_mse_loss.sum() / (non_zero_mask.sum() * self.non_zero_weight + zero_mask.sum() * self.zero_weight)
        weighted_mae_loss = weighted_mae_loss.sum() / (non_zero_mask.sum() * self.non_zero_weight + zero_mask.sum() * self.zero_weight)

        return weighted_mse_loss + weighted_mae_loss



# Training loop function
def training_loop(ddpm, loader, n_epochs, optim, device, display=False, store_path="ddpm_ocean_model.pt"):
    custom_loss = CustomLoss()
    best_loss = float("inf")
    n_steps = ddpm.n_steps

    for epoch in tqdm(range(n_epochs), desc="Training progress", colour="#00ff00"):
        epoch_loss = 0.0
        ddpm.train()
        for step, batch in enumerate(tqdm(loader, leave=False, desc=f"Epoch {epoch + 1}/{n_epochs}", colour="#005500")):
            x0 = batch[0].to(device).float()
            n = len(x0)

        #    if epoch == 0:
                #show_images(x0.cpu(), "Original images before noising")
                #tensors_to_png.generate_png(x0)

            eta = torch.randn_like(x0).to(device)  # Generate noise
            t = torch.randint(0, n_steps, (n,)).to(device)  # Random time steps

            noisy_imgs = ddpm(x0, t, eta)

            eta_theta = ddpm.backward(noisy_imgs, t.reshape(n, -1))

            loss = custom_loss(eta_theta, eta, x0)
            optim.zero_grad()
            loss.backward()
            optim.step()

            epoch_loss += loss.item() * len(x0) / len(loader.dataset)


        if display:
            show_images(generate_new_images(ddpm, device=device), f"Images generated at epoch {epoch + 1}")

        log_string = f"Loss at epoch {epoch + 1}: {epoch_loss:.3f}"

        # Save the best model
        if best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save(ddpm.state_dict(), store_path)
            log_string += " --> Best model ever (stored)"

        print(log_string)


# Create optimizer
optimizer = Adam(ddpm.parameters(), lr=lr)

# Run training loop if not skipping training
if not no_train:
    training_loop(ddpm, loader, n_epochs, optim=optimizer, device=device, store_path=store_path)
