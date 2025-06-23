import os.path
import matplotlib.pyplot as plt

from scipy.io import loadmat
import numpy as np



pycharm_dumb_flag = False
# pickle file is found in the project directory if you run it in pycharm, otherwise it might be in your desktop!

if(os.path.exists("../data/rams_head/stjohn_hourly_5m_velocity_ramhead_v2.mat")):
    pycharm_dumb_flag = True
    file_name = "../data/rams_head/stjohn_hourly_5m_velocity_ramhead_v2.mat"
    mat_data = loadmat(file_name)
else :
    file_name = "data/rams_head/stjohn_hourly_5m_velocity_ramhead_v2.mat"
    mat_data = loadmat(file_name)

print("splitting:", file_name)

u_tensors = mat_data['u']
v_tensors = mat_data['v']

u_tensors = np.expand_dims(u_tensors,axis=2)
v_tensors = np.expand_dims(v_tensors,axis=2)

all_tensors = np.append(u_tensors,v_tensors,2)



# New stuff: analyzing entire data set

# Transpose to shape (T, 2, H, W)
all_tensors = all_tensors.transpose(3, 2, 0, 1)

# Compute mean and variance per time step
means = np.nanmean(all_tensors, axis=(2, 3))  # shape: (T, 2)
vars = np.nanvar(all_tensors, axis=(2, 3))    # shape: (T, 2)

timesteps = np.arange(all_tensors.shape[0])

# Plot mean
plt.figure(figsize=(10, 4))
plt.plot(timesteps, means[:, 0], label='U Mean', color='blue')
plt.plot(timesteps, means[:, 1], label='V Mean', color='green')
plt.title("Mean Value of U/V over Time")
plt.xlabel("Timestep")
plt.ylabel("Mean")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("overall_mean_plot.png")
plt.close()
print("Saved mean plot to overall_mean_plot.png")

# Plot variance
plt.figure(figsize=(10, 4))
plt.plot(timesteps, vars[:, 0], label='U Variance', color='orange')
plt.plot(timesteps, vars[:, 1], label='V Variance', color='red')
plt.title("Variance of U/V over Time")
plt.xlabel("Timestep")
plt.ylabel("Variance")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("overall_variance_plot.png")
plt.close()
print("Saved variance plot to overall_variance_plot.png")

# Print overall stats
print("Overall Mean (U, V):", np.nanmean(means, axis=0))
print("Overall Variance (U, V):", np.nanmean(vars, axis=0))