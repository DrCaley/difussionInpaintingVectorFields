import math
from IPython.display import Image
import random
import imageio
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda

from dataloaders.dataloader import OceanImageDataset
from medium_ddpm.dir.ddpm import MyDDPM
from medium_ddpm.dir.resize import ResizeTransform
from medium_ddpm.dir.unet_resized_v0 import MyUNet
from medium_ddpm.dir.utils import show_images, generate_new_images

# Setting reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Parameters
n_steps, min_beta, max_beta = 1000, 1e-4, 0.02
store_path = "../../../models/ddpm_ocean_v0.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

best_model = MyDDPM(MyUNet(), n_steps=n_steps, device=device)
best_model.load_state_dict(torch.load(store_path, map_location=device))
best_model.eval()
print("Model loaded")


# Define unnormalize function
def unnormalize(tensor):
    return (tensor + 1) / 2

# Generate new images
print("Generating new images")
generated = generate_new_images(
    best_model,
    n_samples=1,
    device=device,
    gif_name="ocean.gif"
)

show_images(generated)

# Initialize dataset with transformations
transform = Compose([
    Lambda(lambda x: (x - 0.5) * 2),  # Normalize to range [-1, 1]
    ResizeTransform((1, 64, 128))  # Resized to (1, 64, 128)
])

store_path = "../../../models/ddpm_ocean_v0.pt"

data = OceanImageDataset(
    mat_file="../../../data/rams_head/stjohn_hourly_5m_velocity_ramhead_v2.mat",
    boundaries="../../../data/rams_head/boundaries.yaml",
    num=1,
    transform=transform
)

loader = DataLoader(data, batch_size=1, shuffle=True)
