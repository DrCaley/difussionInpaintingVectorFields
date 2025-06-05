import torch
import numpy as np
from matplotlib import pyplot as plt
from skimage import data
from skimage.color import rgb2gray
import gpytorch
from noising_process.simple_gp.simple_gp_model import GPModel

# DATA PREP
image = rgb2gray(data.astronaut())
image = image[::4, ::4]  # Downsize for simplicity
original_image = image.copy()

np.random.seed(0)
missing_rate = 0.5
mask = np.random.rand(*image.shape) > missing_rate
known_pixels = mask
unknown_pixels = ~mask
image[unknown_pixels] = np.nan

plt.imshow(image, cmap='gray')
plt.title("Image with Missing Pixels")
plt.show()

coords = np.column_stack(np.nonzero(known_pixels))
values = image[known_pixels]
coords = torch.tensor(coords, dtype=torch.float32)
values = torch.tensor(values, dtype=torch.float32)

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = GPModel(coords, values, likelihood)

# MODEL TRAINING
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


# MODEL USE
unknown_coords = np.column_stack(np.nonzero(unknown_pixels))
unknown_coords = torch.tensor(unknown_coords, dtype=torch.float32)

model.eval()
likelihood.eval()

with torch.no_grad():
    pred = model(unknown_coords)
    pred_values = pred.mean.numpy()

image[unknown_pixels] = pred_values

plt.imshow(image, cmap='gray')
plt.title("Reconstructed Image")
plt.show()
