from random import sample
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from yolo_net_64x128 import Net
from utils.resize_tensor import resize
from utils.tensors_to_png import generate_png
from utils.image_noiser import generate_noised_tensor_iterative


test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

test_dataloader = DataLoader(test_data, batch_size=15, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Net().to(device)
model_path = 'models/mnist_model_ep_1.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


T = 100
use_known_samples = False


def generate_samples(model, T, device, noised_samples, mask):
    sample = noised_samples.clone().float().to(device)
    for t in range(T):
        with torch.no_grad():
            predicted_noise = model(sample, mask)
            sample = sample - predicted_noise
            sample = sample * (1 - mask)
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

    # Verify these are correct
    known_mask = torch.zeros_like(tensor)
    for (y, x) in known_indices:
        known_mask[:, :, y, x] = 1.0

    known_samples = tensor * known_mask

    return known_samples, known_mask


for num, (tensor, _) in enumerate(test_dataloader):
    val_tensor = tensor.to(device).float()
    num_known_points = 340

    if use_known_samples:
        known_samples, known_mask = get_known_samples(val_tensor, num_known_points)
        known_samples = resize(known_samples, (2, 64, 128)).to(device)
        known_mask = resize(known_mask, (2, 64, 128)).to(device)
    else:
        known_samples = resize(val_tensor, (2, 64, 128)).to(device)
        known_mask = torch.ones_like(known_samples).to(device)


    land_mask = torch.zeros_like(known_samples).to(device)

    land_mask[:, :, 10:20, 10:20] = 1

    noised_samples = generate_noised_tensor_iterative(known_samples, T, variance=0.005).float().to(device)

    print(f"Starting to sample {num}/{len(test_dataloader)} number of validation set")
    samples = generate_samples(model, T, device, noised_samples, land_mask)

    generate_png(samples.cpu(), scale=9, output_path="./results", filename=f"output_{num}.png")

    break
