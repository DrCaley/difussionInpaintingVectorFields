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
    Fills in missing regions of a 2D vector field using Gaussian Process regression.

    Args:
        input_image: Tensor of shape [1, 2, H, W]
        mask: Tensor of shape [1, 1, H, W] with 1 for known, 0 for missing
        device: Device to run on

    Returns:
        filled: Tensor of shape [1, 2, H, W]
    """
    input_image = input_image.to(device)
    mask = mask.to(device)

    _, C, H, W = input_image.shape
    input_image = input_image[0]  # [2, H, W]
    mask = mask[0, 0]             # [H, W]

    mask = mask.bool()            # Convert to boolean mask
    mask_flat = mask.flatten()    # [H*W]

    # Coordinates of all pixels on device
    coords = torch.stack(torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    ), dim=-1).reshape(-1, 2).float()  # [H*W, 2]

    filled = input_image.clone()

    for ch in range(C):
        channel_data = input_image[ch].flatten()
        known_values = channel_data[mask_flat]
        if known_values.numel() == 0:
            continue

        known_coords_ch = coords[mask_flat]

        # Prepare tensors for training
        known_coords_ch = known_coords_ch.detach().clone().to(torch.float32).to(device).requires_grad_(True)
        known_values = known_values.detach().clone().to(torch.float32).to(device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        model = GPModel_2D(known_coords_ch, known_values, likelihood).to(device)

        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for i in range(50):
            optimizer.zero_grad()
            output = model(known_coords_ch)
            loss = -mll(output, known_values)

            print(f"Step {i} — loss: {loss.item()}")
            loss.backward()
            optimizer.step()

        model.eval()
        likelihood.eval()
        missing_coords = coords[~mask_flat]

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_dist = likelihood(model(missing_coords))
            preds = pred_dist.mean

        filled[ch].view(-1)[~mask_flat] = preds

    return filled.unsqueeze(0)  # [1, 2, H, W]
