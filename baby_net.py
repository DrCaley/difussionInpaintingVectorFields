import random

import torch
from torch import nn, optim
from image_noiser import generate_noised_tensor_iterative, generate_noised_tensor_single_step
from dataloader import OceanImageDataset
from torch.utils.data import DataLoader


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
        # logits = self.linear_relu_stack(x)
        lin1_output = self.linear1(x)
        sig1 = self.sigmoid(lin1_output)
        lin2_output = self.linear2(sig1)
        sig2 = self.sigmoid(lin2_output)
        lin3_output = self.linear3(sig2)

        return lin3_output


training_data = OceanImageDataset(
    mat_file="./data/rams_head/stjohn_hourly_5m_velocity_ramhead_v2.mat",
    boundaries="./data/rams_head/boundaries.yaml",
    tensor_dir="./data/tensors/"
)

train_dataloader = DataLoader(
    training_data,
    batch_size=1000,
    shuffle=True
)

model = NeuralNetwork().float()

loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)


# Training loop
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    for batch, (tensor, label) in enumerate(dataloader):
        # noise here
        target_iteration = 100

        # less noiser target
        target = generate_noised_tensor_single_step(tensor, target_iteration=random.randint(1, 1000), var_per_iteration=0.005)
        # noiser
        tensor = generate_noised_tensor_iterative(target, iteration=1, variance=0.005)

        target = target.view(target.size(0), -1)
        tensor = tensor.view(tensor.size(0), -1)

        # Compute prediction and loss
        pred = model(target.float())
        loss = loss_fn(pred, tensor.float())

        # Backpropagation
        optimizer.zero_grad()   # clears the accumulated gradients from the prev iteration
        loss.backward()         # computes the gradients of the loss with respect to the model parameters
        optimizer.step()        # updates the model parameters using the computed gradients

        if batch % 10 == 0:
            loss_value, current = loss.item(), batch * len(target)
            print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")


# Test loop
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for target, tensor in dataloader:
            target = target.view(target.size(0), -1)
            tensor = tensor.view(tensor.size(0), -1)
            pred = model(target.float())
            test_loss += loss_fn(pred, tensor.float()).item()
    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")


# Main training and testing loop
epochs = 100
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
print("Done!")
