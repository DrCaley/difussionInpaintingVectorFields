from random import sample
import torch
from torch.utils.data import DataLoader, random_split
import math
from dataloaders.dataloader import OceanImageDataset
from yolo_net_64x128 import Net
from utils.resize_tensor import resize
from utils.tensors_to_png import generate_png
from utils.eval import evaluate
from utils.image_noiser import generate_noised_tensor_iterative

data = OceanImageDataset(
    mat_file="./data/rams_head/stjohn_hourly_5m_velocity_ramhead_v2.mat",
    boundaries="./data/rams_head/boundaries.yaml",
    num=10
)

train_len = int(math.floor(len(data) * 0.7))
test_len = int(math.floor(len(data) * 0.15))
val_len = len(data) - train_len - test_len

torch.manual_seed(42)
training_data, test_data, validation_data = random_split(data, [train_len, test_len, val_len])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 15

train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)
val_loader = DataLoader(validation_data, batch_size=batch_size)

model = Net().to(device)
model_path = 'models/yolo_model_epoch_150.pth'
model.eval()

T = 1


def generate_samples(model, T, device, noised_samples, mask):
    sample = noised_samples.clone().float().to(device)

    for t in range(T):
        with torch.no_grad():
            predicted_noise = model(sample, mask)
            sample = sample - predicted_noise

    return sample


def get_known_samples(tensor, num_known_points):
    b, c, h, w = tensor.shape
    total_points = h * w

    if num_known_points > total_points:
        raise ValueError("num_known_points is greater than the total number of points in the tensor")

    ocean_indices = [(y, x) for y in range(h) for x in range(w) if tensor[:, :, y, x].sum() == 0]

    if num_known_points > len(ocean_indices):
        raise ValueError("num_known_points is greater than the number of ocean points")

    known_indices = sample(ocean_indices, num_known_points)

    known_mask = torch.zeros_like(tensor)
    for (y, x) in known_indices:
        known_mask[:, :, y, x] = 1.0

    known_samples = tensor * known_mask

    return known_samples, known_mask


for num, (tensor, _) in enumerate(val_loader):
    if num == 0:
        val_tensor = tensor.to(device).float()
        num_known_points = 340

        known_samples, known_mask = get_known_samples(val_tensor, num_known_points)
        land_mask = (tensor != 0).float()

        known_samples = resize(known_samples, (2, 64, 128)).to(device)
        known_mask = resize(known_mask, (2, 64, 128)).to(device)
        land_mask = resize(land_mask, (2, 64, 128)).to(device)

        noised_samples = generate_noised_tensor_iterative(known_samples, T, variance=0.005).float().to(device)

        print(f"Starting to sample {num}/{len(val_loader)} number of validation set")
        samples = generate_samples(model, T, device, noised_samples, land_mask)

        for i, sample in enumerate(samples):
            generate_png(sample, scale=9)

        break
