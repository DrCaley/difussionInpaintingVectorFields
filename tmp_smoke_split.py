"""Quick forward/backward test of MyUNet_Helmholtz_Split."""
import torch
from ddpm.neural_networks.unets.unet_helmholtz_split import MyUNet_Helmholtz_Split

net = MyUNet_Helmholtz_Split(n_steps=250)
x = torch.randn(2, 2, 64, 128)
t = torch.randint(0, 250, (2,))
out = net(x, t)
print(f"Input:  {x.shape}")
print(f"Output: {out.shape}")
print(f"v_sol:  {net.last_v_sol.shape}, v_irr: {net.last_v_irr.shape}")
print(f"psi:    {net.last_psi.shape}, phi: {net.last_phi.shape}")
loss = out.sum()
loss.backward()
grad_norms = {n: p.grad.norm().item() for n, p in net.named_parameters() if p.grad is not None}
psi_grads = {k: v for k, v in grad_norms.items() if "psi" in k}
phi_grads = {k: v for k, v in grad_norms.items() if "phi" in k}
shared = {k: v for k, v in grad_norms.items() if "psi" not in k and "phi" not in k}
print(f"Params with grads: {len(grad_norms)}")
print(f"PSI branch params: {len(psi_grads)}, PHI branch params: {len(phi_grads)}, Shared params: {len(shared)}")
print(f"Total params: {sum(p.numel() for p in net.parameters()):,}")
print("OK - forward + backward pass succeeded")
