import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import gpytorch
from torch.utils.data import DataLoader
from dataloaders.dataloader import OceanImageDataset
from gaussian_process.simple_gp.simple_gp import prepare_data, train_gp_model, predict_gp_model
from gaussian_process.simple_gp.simple_gp_model import GPModel_2D

# TODO: Make script to test gp model against ddpm on similar masks
# TODO: Test if can make from OceanImageDataset
data = OceanImageDataset(num=1)
train_loader = DataLoader(data, batch_size=1, shuffle=True)
image = train_loader.dataset[0][0]

np.random.seed(0)
missing_rate = 0.10
missing_pixel_mask = np.random.rand(*image.shape[:2]) > missing_rate

masked_image = image.clone()
masked_image[missing_pixel_mask] = np.nan

plt.imshow(np.nan_to_num(masked_image / 255, nan=1.0))
plt.title("Image with Missing Pixels")
plt.show()

# Prep
coords, values_r = prepare_data(0)
_, values_g = prepare_data(1)

likelihood_r = gpytorch.likelihoods.GaussianLikelihood()
model_r = GPModel_2D(coords, values_r, likelihood_r)

likelihood_g = gpytorch.likelihoods.GaussianLikelihood()
model_g = GPModel_2D(coords, values_g, likelihood_g)

# Training
train_gp_model(model_r, likelihood_r, coords, values_r)
train_gp_model(model_g, likelihood_g, coords, values_g)

unknown_coords = np.column_stack(np.where(missing_pixel_mask))
unknown_coords = torch.tensor(unknown_coords, dtype=torch.float32)

# Predictions
pred_r = predict_gp_model(model_r, likelihood_r, unknown_coords)
pred_g = predict_gp_model(model_g, likelihood_g, unknown_coords)

masked_image[..., 0][missing_pixel_mask] = pred_r
masked_image[..., 1][missing_pixel_mask] = pred_g

masked_image = np.nan_to_num(masked_image, nan=0.0)

output_image = Image.fromarray(masked_image.astype(np.uint8))
output_image.save('./gp_image.png')
