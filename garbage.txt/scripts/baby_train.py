import math
import random
import torch
from torch import nn, optim
from utils.image_noiser import generate_noised_tensor_iterative, generate_noised_tensor_single_step
from dataloaders.dataloader import OceanImageDataset
from torch.utils.data import DataLoader, random_split
import setproctitle

setproctitle.setproctitle("ocean_motion_model_training")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(3 * 44 * 94, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 3 * 44 * 94)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.flatten(x)
        lin1_output = self.linear1(x)
        sig1 = self.sigmoid(lin1_output)
        lin2_output = self.linear2(sig1)
        sig2 = self.sigmoid(lin2_output)
        lin3_output = self.linear3(sig2)
        return lin3_output


data = OceanImageDataset(
    mat_file="./data/rams_head/stjohn_hourly_5m_velocity_ramhead_v2.mat",
    boundaries="./data/rams_head/boundaries.yaml",
    num=1000
)

train_len = int(math.floor(len(data) * 0.7))
test_len = int(math.floor(len(data) * 0.15))
val_len = len(data) - train_len - test_len

training_data, test_data, validation_data = random_split(data, [train_len, test_len, val_len])

train_dataloader = DataLoader(
    training_data,
    batch_size=5,
    shuffle=True
)

val_dataloader = DataLoader(
    validation_data,
    batch_size=5,
    shuffle=False
)

test_dataloader = DataLoader(
    test_data,
    batch_size=5,
    shuffle=False
)

model = NeuralNetwork().to(device).float()

loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    for batch, (tensor, label) in enumerate(dataloader):
        target_iteration = 100

        # Generate noisy data
        target = generate_noised_tensor_single_step(tensor, target_iteration=random.randint(1, 1000),
                                                    var_per_iteration=0.005)
        tensor = generate_noised_tensor_iterative(target, iteration=1, variance=0.005)

        target = target.view(target.size(0), -1).to(device)
        tensor = tensor.view(tensor.size(0), -1).to(device)

        # Compute prediction and loss
        pred = model(target.float())
        loss = loss_fn(pred, tensor.float())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss_value, current = loss.item(), batch * len(target)
            print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")


# Main training and testing loop
epochs = 1000
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
