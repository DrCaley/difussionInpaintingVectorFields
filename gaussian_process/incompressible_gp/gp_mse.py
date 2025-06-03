import csv
import math
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from DataPrep.ocean_image_dataset import OceanImageDataset

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

mse_results = pd.read_csv('./mse_results.csv')

mse_u = mse_results['mse_u']
mse_v = mse_results['mse_v']

tot_mse = mse_u + mse_v
avg_mse = tot_mse.mean()

print(f"Total average MSE: {avg_mse}")

# data = OceanImageDataset(num=17040)
# train_len = int(math.floor(len(data) * 0.7))
# test_len = int(math.floor(len(data) * 0.15))
# val_len = len(data) - train_len - test_len
#
# training_data, test_data, validation_data = random_split(data, [train_len, test_len, val_len])
#
# batch_size = 1
# train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_data, batch_size=batch_size)
# val_loader = DataLoader(validation_data, batch_size=batch_size)
#
# with open('test_index_17040.csv', 'w', newline='') as csvfile:
#     fieldnames = ['world_idx']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#
#     writer.writeheader()
#     for batch in test_loader:
#         idx = batch[1][0].item()
#         writer.writerow({'world_idx': idx})
