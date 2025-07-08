import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSV
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'mse_vs_mask_coverage_weekend_ddpm_ocean_model.csv')
data = pd.read_csv(csv_path)

# Group by Mask Percentage and collect MSEs
grouped_ddpm = data.groupby('Mask Percentage')['MSE_DDPM'].apply(list)
grouped_gp = data.groupby('Mask Percentage')['MSE_GP'].apply(list)

# Sort by Mask Percentage
sorted_distances = sorted(grouped_ddpm.index)
ddpm_means = [np.mean(grouped_ddpm[dist]) for dist in sorted_distances]
ddpm_stds = [np.std(grouped_ddpm[dist]) for dist in sorted_distances]

gp_means = [np.mean(grouped_gp[dist]) for dist in sorted_distances]
gp_stds = [np.std(grouped_gp[dist]) for dist in sorted_distances]

# X positions
num_groups = len(sorted_distances)
positions_ddpm = np.array([i * 2 for i in range(num_groups)])
positions_gp = positions_ddpm + 0.8

# Plot
plt.figure(figsize=(10, 6))
bar_width = 0.6

# Plot bars with error bars
plt.bar(positions_ddpm, ddpm_means, yerr=ddpm_stds, width=bar_width, color='blue', label='MSE_DDPM', capsize=5)
plt.bar(positions_gp, gp_means, yerr=gp_stds, width=bar_width, color='red', label='MSE_GP', capsize=5)

# X-ticks centered between bar pairs
xtick_positions = (positions_ddpm + positions_gp) / 2
xtick_labels = [f"{d:.1f}" for d in sorted_distances]
plt.xticks(xtick_positions, xtick_labels, rotation=45)

plt.xlabel('Mask Percentage')
plt.ylabel('MSE')
plt.title('MSE Comparison: DDPM vs GP by Mask Percentage')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('mse_mask_comparison_errorbar.png', dpi=300)
plt.show()