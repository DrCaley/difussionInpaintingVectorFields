import os
import pickle
import random
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def get_dd_initializer():
    from data_prep.data_initializer import DDInitializer
    return DDInitializer()

batches_of_noise = 10
timesteps_per_batch = 100
num_samples = 10
total_timesteps = batches_of_noise * timesteps_per_batch

noise_path = "noise"
height = 64
width = 128
batch_size = 1
device = get_dd_initializer().get_device()
os.makedirs(noise_path, exist_ok=True)

print(f"creating noise for "
      f"{total_timesteps} timesteps in {noise_path}")

from noising_process.incompressible_gp.adding_noise.divergence_free_noise import generate_div_free_noise
samples = []
for _ in range(total_timesteps):
    t_noise_samples = []
    for i in range(num_samples):
        noise = generate_div_free_noise(batch_size, height, width, device)
        if i - 1 > 0:
            noise += t_noise_samples[i - 1]
        t_noise_samples.append(noise)
    samples.append(t_noise_samples)

for noise_batch in range(batches_of_noise):
    start_idx = noise_batch * timesteps_per_batch
    end_idx = (noise_batch + 1) * timesteps_per_batch
    filename = f"{start_idx + 1}-{end_idx}.pickle"

    with open(os.path.join(noise_path, filename), 'wb') as f:
        pickle.dump(samples[start_idx:end_idx], f)

print("You have been pickled")