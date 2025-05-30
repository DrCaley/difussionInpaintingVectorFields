from IPython.display import Image
import random
import numpy as np
import torch

from medium_ddpm.ddpm import MyDDPM
from medium_ddpm.dir.unets.unet_resized_2_channel_xl import MyUNet
from medium_ddpm.dir.util import show_images, generate_new_images

# Setting reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Parameters
n_steps, min_beta, max_beta = 1000, 1e-4, 0.02
store_path = "../../../DDPM/Trained_Models/ddpm_ocean_2channel.pt"

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
    gif_name="ocean.gif",
    c=2
)
show_images(generated, "Final result")

Image(open('../../../DDPM/Helper_Functions/ocean.gif', 'rgb').read())
