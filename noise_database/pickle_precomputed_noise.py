import os
import pickle
import random

base_dir = os.path.dirname(os.path.abspath(__file__))
from data_prep.data_initializer import DDInitializer
from noising_process.incompressible_gp.adding_noise.divergence_free_noise import \
    generate_div_free_noise, layered_div_free_noise

dd = DDInitializer

timesteps: int = 10
noise_path = "noise"
height = 100
width = 100
batch_size = 1
num_samples = 10
device = dd.get_device()
os.makedirs(noise_path, exist_ok=True)

print(f"creating {noise_strat} noise for "
      f"{timesteps} timesteps in {noise_path}")

samples = []
for _ in range(timesteps):
    t_noise_samples = []
    for i in range(num_samples):

    samples.append(t_noise_samples)

with open(f"", 'wb') as f:
    pickle.dump(samples, f)

print("You have been pickled")