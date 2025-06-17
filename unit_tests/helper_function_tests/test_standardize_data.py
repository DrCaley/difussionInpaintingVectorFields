import os
import sys
import csv
import torch
from math import isclose
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from ddpm.helper_functions.standardize_data import ZScoreStandardizer, MaxMagnitudeStandardizer, UnitVectorNormalizer
from plots.plot_vector_field_tool import plot_vector_field
from data_prep.data_initializer import DDInitializer
from ddpm.helper_functions.compute_divergence import compute_divergence



""" Uncomment below to get csv file: 
used for testing if sign of divergence changes after z transform.
Currently believe sign never changes, but monotonicity IS lost
(more div initially may not have more div afterwards).

# CSV
csv_file ="divergences_test.csv"
with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Number', 'Divergence OG', 'Divergence Z Transformed'])

for i in range(50000):
    u = torch.randn(1, 94)
    v = torch.randn(1, 94)
    original_tensor = torch.stack([u, v], dim=0)

    # Compute stats
    u_mean = original_tensor[0].mean()
    v_mean = original_tensor[1].mean()
    u_std = original_tensor[0].std()
    v_std = original_tensor[1].std()

    # Avoid division by zero if std == 0
    if u_std == 0:
        u_std = torch.tensor(1.0)
    if v_std == 0:
        v_std = torch.tensor(1.0)

    z_score_standardizer = ZScoreStandardizer(u_mean, u_std, v_mean, v_std)

    z_score_tensor = z_score_standardizer(original_tensor)

    a = compute_divergence(original_tensor[0], original_tensor[1]).nanmean().item()
    b = compute_divergence(z_score_tensor[0], z_score_tensor[1]).nanmean().item()
    with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([i + 1, a, b])
"""

""" Uncomment below to test max_mag standardization: 
Currently think it works perfectly.

u = 0.25 * torch.randn(1, 94)
v = 0.25 * torch.randn(1, 94)
original_tensor = torch.stack([u, v], dim=0)
plot_vector_field(original_tensor[0], original_tensor[1], file="vector_field_unchanged.png")
a = compute_divergence(original_tensor[0], original_tensor[1]).nanmean().item()

max_mag_standardizer = MaxMagnitudeStandardizer()
max_mag_tensor = max_mag_standardizer(original_tensor)
c = max_mag_standardizer.get_max_magnitude(original_tensor).item()
plot_vector_field(max_mag_tensor[0], max_mag_tensor[1], file="vector_field_max_mag.png")
b = compute_divergence(max_mag_tensor[0], max_mag_tensor[1]).nanmean().item()

# Scaling field down by a constant should scale divergence by same constant.
assert isclose(a, b * c, rel_tol=1e-5), "Scaling is incorrect."

# The largest magnitude after applying scaling should be 1.
assert isclose(max_mag_standardizer.get_max_magnitude(max_mag_tensor).item(), 1, rel_tol=1e-5), "Largest magnitude is not 1."

inverted_tensor = max_mag_standardizer.unstandardize(max_mag_tensor)
plot_vector_field(inverted_tensor[0], inverted_tensor[1], file="vector_field_max_mag_inverted.png")
"""

u = 0.25 * torch.randn(1, 94)
v = 0.25 * torch.randn(1, 94)
original_tensor = torch.stack([u, v], dim=0)
plot_vector_field(original_tensor[0], original_tensor[1], scale=2, file="vector_field_unchanged.png")
a = compute_divergence(original_tensor[0], original_tensor[1]).nanmean().item()

unit_vector_standardizer = UnitVectorNormalizer()
normalized_tensor = unit_vector_standardizer(original_tensor)
plot_vector_field(normalized_tensor[0], normalized_tensor[1], scale=2, file="vector_field_normed.png")
b = compute_divergence(normalized_tensor[0], normalized_tensor[1]).nanmean().item()

# Test that these are indeed all unit vectors
magnitudes = torch.sqrt(normalized_tensor[0]**2 + normalized_tensor[1]**2)
assert torch.allclose(magnitudes, torch.ones_like(magnitudes), atol=1e-5), "❌ Some magnitudes are not close to 1"

inverted_tensor = unit_vector_standardizer.unstandardize(normalized_tensor)

# Test that divergence is the same after normalize followed by unnormalize
c = compute_divergence(inverted_tensor[0], inverted_tensor[1]).nanmean().item()
assert abs(a - c) < 1e-5, "❌ Divergence not preserved after unstandardization"


plot_vector_field(inverted_tensor[0], inverted_tensor[1], scale=2, file="vector_field_normed_inverted.png")

