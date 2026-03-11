"""Quick validation of HelmholtzSupervisionLoss."""
import torch
from ddpm.helper_functions.loss_functions import HelmholtzSupervisionLoss

loss_fn = HelmholtzSupervisionLoss(lambda_decomp=0.1, lambda_orth=0.01, warmup_epochs=0)
print("Loss created OK")
print(f"  lambda_decomp={loss_fn.lambda_decomp}, lambda_orth={loss_fn.lambda_orth}")

class FakeNet:
    last_v_sol = torch.randn(4, 2, 64, 128)
    last_v_irr = torch.randn(4, 2, 64, 128)

class FakeDDPM:
    n_steps = 250
    network = FakeNet()

pred = torch.randn(4, 2, 64, 128)
target = torch.randn(4, 2, 64, 128)
noisy = torch.randn(4, 2, 64, 128)
x0 = torch.randn(4, 2, 64, 128)
t = torch.tensor([5, 10, 200, 3])

loss = loss_fn(pred, target, noisy, epoch=5, ddpm=FakeDDPM(), x0=x0, t=t, prediction_target="x0")
print(f"Loss value: {loss.item():.6f}")

# Also check warmup gating returns base only
loss_fn2 = HelmholtzSupervisionLoss(lambda_decomp=0.1, lambda_orth=0.01, warmup_epochs=10)
loss2 = loss_fn2(pred, target, noisy, epoch=3, ddpm=FakeDDPM(), x0=x0, t=t, prediction_target="x0")
print(f"Warmup-gated loss: {loss2.item():.6f} (should be pure MSE)")

print("\nAll OK!")
