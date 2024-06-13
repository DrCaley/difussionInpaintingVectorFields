import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from random import randint

from dataloader import OceanImageDataset
from example_nn.evaluation import evaluate
import math
from example_nn.inPaintingNetwork import Net

from resize_tensor import resize
from image_noiser import generate_noised_tensor_single_step, generate_noised_tensor_iterative
from tensors_to_png import generate_png


class Args:
    lr = 0.001
    lr_finetune = 0.0001
    resume = None
    max_iter = 10000
    log_interval = 100
    save_model_interval = 1000
    vis_interval = 500
    save_dir = "./checkpoints"
    finetune = False
    LAMBDA_DICT = {
        'valid': 1.0,
        'hole': 6.0,
        'prc': 0.05,
        'style': 120.0,
        'tv': 0.1
    }


args = Args()
data = OceanImageDataset(
    mat_file="./data/rams_head/stjohn_hourly_5m_velocity_ramhead_v2.mat",
    boundaries="./data/rams_head/boundaries.yaml",
    num=100
)

train_len = int(math.floor(len(data) * 0.7))
test_len = int(math.floor(len(data) * 0.15))
val_len = len(data) - train_len - test_len

training_data, test_data, validation_data = random_split(data, [train_len, test_len, val_len])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader = DataLoader(training_data, batch_size=1, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1)
val_loader = DataLoader(validation_data, batch_size=1)

print(f"Number of training samples: {len(training_data)}")

model = Net().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

start_iter = 0
num_epochs = args.max_iter // len(train_loader) + 1

for epoch in range(num_epochs):
    model.train()
    for i, (tensor, label) in enumerate(tqdm(train_loader)):
        tensor = tensor.float()

        # Generate mask (assuming the mask is 1 where the data is missing and 0 elsewhere)
        mask = (tensor != 0).float()

        target = generate_noised_tensor_single_step(tensor, target_iteration=randint(1, 1000),
                                                    var_per_iteration=0.005).float()
        tensor = generate_noised_tensor_iterative(target, iteration=1, variance=0.005).float()

        tensor = resize(tensor, (3, 512, 512))
        target = resize(target, (3, 512, 512))
        mask = resize(mask, (3, 512, 512))
        # Forward pass through the model
        output = model(tensor, mask)

        #calculate loss
        output = output * mask
        target = target * mask
        squared_error = (output - target) ** 2
        masked_squared_error = squared_error * mask
        loss = masked_squared_error.sum() / mask.sum()

        if i % 100 == 0:
            print(loss.item())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch * len(train_loader) + i + 1) % args.vis_interval == 0:
            model.eval()
            evaluate(model, test_loader, device,
                     '{:s}/images/test_{:d}.jpg'.format(args.save_dir, epoch * len(train_loader) + i + 1))


