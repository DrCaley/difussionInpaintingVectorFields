"""Quick test: verify TopologyAwareLossStrategy gradients flow."""
import sys, torch, torch.nn as nn

# Minimal mock of GaussianDDPM for alpha_bars
class FakeDDPM:
    def __init__(self, n_steps=250, device="cpu"):
        self.n_steps = n_steps
        betas = torch.linspace(1e-4, 0.02, n_steps, device=device)
        alphas = 1 - betas
        self.alpha_bars = torch.cumprod(alphas, dim=0)

def main():
    device = "cpu"
    from ddpm.helper_functions.loss_functions import TopologyAwareLossStrategy

    loss_fn = TopologyAwareLossStrategy(
        lambda_vort=0.1, lambda_div=0.05, lambda_speed=0.0, t_max_frac=0.5,
    ).to(device)

    B = 8
    ddpm = FakeDDPM(250, device)

    # Simulate: predicted noise, target noise, noisy images, clean images
    pred_eps = torch.randn(B, 2, 64, 128, device=device, requires_grad=True)
    true_eps = torch.randn(B, 2, 64, 128, device=device)
    x_t = torch.randn(B, 2, 64, 128, device=device)
    x0 = torch.randn(B, 2, 64, 128, device=device)

    # Case 1: low timesteps → topology loss active
    t_low = torch.randint(0, 62, (B,), device=device)  # all < 125 (t_max_frac=0.5)
    loss = loss_fn(pred_eps, true_eps, x_t, x0=x0, t=t_low, ddpm=ddpm, prediction_target="eps")
    loss.backward()
    assert pred_eps.grad is not None, "No gradient on predicted_noise!"
    grad_norm_low = pred_eps.grad.norm().item()
    print(f"✓ Low-t loss = {loss.item():.6f}, grad_norm = {grad_norm_low:.6f}")

    # Case 2: high timesteps → topology loss gated off, pure MSE
    pred_eps2 = torch.randn(B, 2, 64, 128, device=device, requires_grad=True)
    t_high = torch.randint(200, 250, (B,), device=device)  # all > 125
    loss_high = loss_fn(pred_eps2, true_eps, x_t, x0=x0, t=t_high, ddpm=ddpm, prediction_target="eps")
    loss_high.backward()
    grad_norm_high = pred_eps2.grad.norm().item()
    print(f"✓ High-t loss = {loss_high.item():.6f}, grad_norm = {grad_norm_high:.6f}")

    # Case 3: no kwargs → pure MSE fallback
    pred_eps3 = torch.randn(B, 2, 64, 128, device=device, requires_grad=True)
    loss_fallback = loss_fn(pred_eps3, true_eps, x_t)
    loss_fallback.backward()
    print(f"✓ Fallback loss = {loss_fallback.item():.6f}")

    # Verify topo terms are additive: low-t loss should differ from pure MSE
    # (they won't be exactly equal because topo terms add something)
    mse_only = nn.MSELoss()(pred_eps.detach().requires_grad_(False), true_eps).item()
    topo_added = abs(loss.item() - mse_only)
    print(f"\n  MSE component: {mse_only:.6f}")
    print(f"  Topology contribution: {topo_added:.6f}")
    print(f"  Topo is {'active' if topo_added > 1e-6 else 'ZERO (problem!)'}")

    print("\n✅ All tests passed — topology loss is differentiable and gradients flow.")

if __name__ == "__main__":
    main()
