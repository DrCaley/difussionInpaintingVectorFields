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
    train=True,
    download=True,
    transform=ToTensor()
)

test_dataloader = DataLoader(test_data, batch_size=5, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
model_path = '../../models/mnist_model_v2_ep_10.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

T = 1000

def generate_samples(model, T, device, noised_samples, mask):
    sample = noised_samples.clone().to(device)
    for t in range(T):
        with torch.no_grad():
            predicted_noise = model(sample, mask)
            sample = sample - predicted_noise
    return sample       # nan at T = 1000


for num, (tensor, _) in enumerate(test_dataloader):
    if num == 0:
        tensor = tensor.to(device).float()
        num_known_points = 100

        known_samples = resize(tensor, (2, 64, 128)).to(device)
        # White (1) for size of actual, black (0) otherwise
        img_size = torch.ones((5, 2, 28, 28))
        known_mask = resize((img_size != 0).float(), (2, 64, 128)).to(device)

        # generate_png(known_samples.cpu(), scale=9, output_path="../../results", filename=f"mnist_sample_{num}.png")
        # generate_png(known_mask.cpu(), scale=9, output_path="../../results", filename=f"mnist_mask_{num}.png")

        noised_samples = generate_noised_tensor_iterative(known_samples, T, variance=0.005).float().to(device)
        generate_png(noised_samples.cpu(), scale=9, output_path="../../results",
                     filename=f"mnist_noised_samples_{num}.png")

        print(f"Starting to sample batch {num}/{len(test_dataloader)} number of validation set")
        samples = generate_samples(model, T, device, noised_samples, known_mask)

        generate_png(samples.cpu(), scale=9, output_path="../../results", filename=f"mnist_output_{num}.png")
        # PIL.Image.fromarray(samples.numpy()[0, 0, :, :] * 255).show()
        break
