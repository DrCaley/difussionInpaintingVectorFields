import torch
import gpytorch

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

device = "cuda" if torch.cuda.is_available() else "cpu"
train_x = torch.rand(20, 2, device=device)
train_y = torch.sin(train_x[:,0]) + torch.cos(train_x[:,1])

likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
model = GPModel_2D(train_x, train_y, likelihood).to(device)

model.train()
likelihood.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(10):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    print(f"Step {i} — loss: {loss.item()}")
    loss.backward()
    optimizer.step()
