import torch
import gpytorch

# Gaussian Process model for 2D vector fields
class GPModel_2D(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=2)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def gp_fill(input_image, mask, device="cuda"):
    """
    input_image: Tensor of shape [1, 2, H, W]
    mask: Tensor of shape [1, 1, H, W] with 1 for known, 0 for missing
    """
    _, C, H, W = input_image.shape
    input_image = input_image[0]     # shape [2, H, W]
    mask = mask[0, 0]                # shape [H, W]

    coords = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij'), dim=-1)  # [H, W, 2]
    coords = coords.reshape(-1, 2).float().to(device)  # [H*W, 2]

    mask_flat = mask.flatten()  # [H*W]
    known_coords = coords[mask_flat.bool()]     # [N_known, 2]
    missing_coords = coords[~mask_flat.bool()]  # [N_missing, 2]

    # Create output tensor
    filled = input_image.clone()

    for ch in range(C):  # For each channel (2 total)
        channel_data = input_image[ch].flatten()
        known_values = channel_data[mask_flat.bool()]  # [N_known]

        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        model = GPModel_2D(known_coords, known_values, likelihood).to(device)

        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for _ in range(50):  # Train GP
            optimizer.zero_grad()
            output = model(known_coords)
            loss = -mll(output, known_values)
            loss.backward()
            optimizer.step()

        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_dist = likelihood(model(missing_coords))
            preds = pred_dist.mean

        # Assign predictions to missing locations
        filled[ch].flatten()[~mask_flat.bool()] = preds

    return filled.unsqueeze(0)  # shape [1, 2, H, W]
