import os
import pandas as pd
import matplotlib.pyplot as plt

# Get the directory this script is in
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'inpainting_xl_data.csv')

# Load CSV
data = pd.read_csv(csv_path)

# Clean MSE and mask_percent columns
data['mse'] = pd.to_numeric(data['mse'], errors='coerce')
data['mask_percent'] = pd.to_numeric(data['mask_percent'], errors='coerce')

# Group by mask percentage
grouped = data.groupby('mask_percent')['mse'].apply(list)

# Prepare box plot data
labels = [f"{pct:.1f}%" for pct in grouped.index]
box_data = grouped.tolist()

# Calculate positions with spacing (1 unit apart)
n = len(box_data)
positions = [i * 1 + 1 for i in range(n)]

# Plot with vertical boxes at spaced positions
plt.figure(figsize=(8, 6))
plt.boxplot(box_data, vert=True, patch_artist=True,
            boxprops=dict(facecolor='lightblue'),
            positions=positions)

# Show every 10th label with adjusted tick positions
tick_indices = list(range(0, len(labels), 5)) # Every 5th percentage label
tick_positions = [positions[i] for i in tick_indices]
tick_labels = [labels[i] for i in tick_indices]

plt.xticks(tick_positions, tick_labels, rotation=45)
plt.title('MSE Distribution by Mask Percentage')
plt.ylabel('MSE')
plt.xlabel('Mask Percentage')
plt.tight_layout()
plt.savefig('mse_by_mask_percentage.png', dpi=300, bbox_inches='tight')
plt.show()