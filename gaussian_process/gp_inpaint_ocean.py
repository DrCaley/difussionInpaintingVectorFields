import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import gpytorch
from gaussian_process.gp_model import GPModel_2D

image = Image.open('imgs/input_img_cropped.png')
image = np.array(image, dtype=np.float32)

mask = Image.open('imgs/mask_cropped.png').convert('L')
mask = np.array(mask, dtype=np.float32)
missing_pixel_mask = mask > 0

masked_image = image.copy()
masked_image[missing_pixel_mask] = np.nan

plt.imshow(np.nan_to_num(masked_image / 255, nan=1.0))
plt.title("Image with Missing Pixels")
plt.show()

def prepare_data(channel):
    coords = np.column_stack(np.where(~missing_pixel_mask))
    values = image[..., channel][~missing_pixel_mask]

    coords = torch.tensor(coords, dtype=torch.float32)
    values = torch.tensor(values, dtype=torch.float32)
    return coords, values

coords, values_r = prepare_data(0)
_, values_g = prepare_data(1)

likelihood_r = gpytorch.likelihoods.GaussianLikelihood()
model_r = GPModel_2D(coords, values_r, likelihood_r)

likelihood_g = gpytorch.likelihoods.GaussianLikelihood()
model_g = GPModel_2D(coords, values_g, likelihood_g)

def train_gp_model(model, likelihood, coords, values):
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    n_iter = 100
    for i in range(n_iter):
        optimizer.zero_grad()
        output = model(coords)
        loss = -mll(output, values)
        loss.backward()
        optimizer.step()
        if (i + 1) % 10 == 0:
            print(f"Iter {i + 1}/{n_iter} - Loss: {loss.item():.3f}")

train_gp_model(model_r, likelihood_r, coords, values_r)
train_gp_model(model_g, likelihood_g, coords, values_g)

unknown_coords = np.column_stack(np.where(missing_pixel_mask))
unknown_coords = torch.tensor(unknown_coords, dtype=torch.float32)

def predict_gp_model(model, likelihood, unknown_coords):
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        pred = model(unknown_coords)
        pred_values = pred.mean.numpy()
    return pred_values

pred_r = predict_gp_model(model_r, likelihood_r, unknown_coords)
pred_g = predict_gp_model(model_g, likelihood_g, unknown_coords)

masked_image[..., 0][missing_pixel_mask] = pred_r
masked_image[..., 1][missing_pixel_mask] = pred_g
# masked_image[..., 2][missing_pixel_mask] = 0

masked_image = np.nan_to_num(masked_image, nan=0.0)

# (44, 94, 3)
output_image = Image.fromarray(masked_image.astype(np.uint8))
output_image.save('./reconstructed_image.png')

expanded_mask = np.repeat(mask[..., np.newaxis], 3, axis=2)

def calculate_mse(original_image, reconstructed_image, mask):
    masked_original = original_image * mask
    masked_reconstructed = reconstructed_image * mask
    mse = np.mean((masked_original - masked_reconstructed) ** 2)
    return mse

# TODO: this MSE is huge
mse_gp = calculate_mse(image, masked_image, expanded_mask)
print(f"GP MSE: {mse_gp}")

ddpm_image = Image.open('imgs/ddpm_img_cropped.png')
ddpm_image = np.array(ddpm_image, dtype=np.float32)
mse_ddpm = calculate_mse(ddpm_image, masked_image, expanded_mask)
print(f"DDPM MSE: {mse_ddpm}")

plt.imshow(masked_image.astype(np.uint8))
plt.title("Reconstructed Image")
plt.show()