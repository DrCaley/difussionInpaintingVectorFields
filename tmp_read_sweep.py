import torch

d = torch.load("results/timestep_sweep/sweep_ens10.pt", map_location="cpu", weights_only=False)
r = d["sweep_results"]
gp = d["summary"]["gp_mean_mse"]
cnn = d["summary"]["gpcnn_mean_mse"]
print(f"GP MSE:  {gp:.6f}")
print(f"CNN MSE: {cnn:.6f} ({cnn/gp:.3f}x GP)")
print()

import statistics

header = f"{'t':>5} | {'Ens MSE':>10} | {'Ens/GP':>7} | {'Comp MSE':>10} | {'Comp/GP':>8} | {'Comp<CNN':>8} | {'Corr':>6}"
print(header)
print("-" * len(header))

for t in [25, 50, 75, 100, 150, 200]:
    samples = r[t]
    ens_mses = [s["ens_mse"] for s in samples]
    comp_mses = [s["comp_mse"] for s in samples]
    cnn_mses = [s["gpcnn_mse"] for s in samples]
    corrs = [s["corr_comp"] for s in samples]
    
    mean_ens = statistics.mean(ens_mses)
    mean_comp = statistics.mean(comp_mses)
    comp_wins = sum(1 for c, n in zip(comp_mses, cnn_mses) if c < n)
    mean_corr = statistics.mean(corrs)
    
    print(f"{t:>5} | {mean_ens:>10.6f} | {mean_ens/gp:>6.3f}x | {mean_comp:>10.6f} | {mean_comp/gp:>7.3f}x | {comp_wins:>5}/100 | {mean_corr:>6.3f}")

print()
print("Eddy subset:")
header2 = f"{'t':>5} | {'Ens MSE':>10} | {'Comp MSE':>10} | {'Comp<CNN':>8}"
print(header2)
print("-" * len(header2))
for t in [25, 50, 75, 100, 150, 200]:
    samples = r[t]
    eddy = [s for s in samples if s["is_eddy"]]
    non_eddy = [s for s in samples if not s["is_eddy"]]
    e_ens = statistics.mean([s["ens_mse"] for s in eddy])
    e_comp = statistics.mean([s["comp_mse"] for s in eddy])
    e_wins = sum(1 for s in eddy if s["comp_mse"] < s["gpcnn_mse"])
    ne_ens = statistics.mean([s["ens_mse"] for s in non_eddy])
    ne_comp = statistics.mean([s["comp_mse"] for s in non_eddy])
    ne_wins = sum(1 for s in non_eddy if s["comp_mse"] < s["gpcnn_mse"])
    print(f"{t:>5} | E:{e_ens:.6f} | E:{e_comp:.6f} | E:{e_wins}/50  NE:{ne_wins}/50")
