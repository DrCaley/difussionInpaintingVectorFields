import os
from random import randint

import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm

from utils.tensors_to_png import generate_png
from yolo_net_64x128 import Net
from utils.resize_tensor import resize
from utils.image_noiser import generate_noised_tensor_single_step, generate_noised_tensor_iterative
from utils.eval import evaluate


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

num_epochs = 1
save_interval = 1

train_losses = []
test_losses = []

for epoch in range(num_epochs):
    model.train()
    for batch, (tensor, label) in enumerate(tqdm(train_dataloader)):
        tensor = tensor.float()
        mask = tensor.float()

        target = generate_noised_tensor_single_step(tensor, target_iteration=randint(1, 1000),
                                                    var_per_iteration=0.005).float().to(device)
        tensor = generate_noised_tensor_iterative(target, iteration=1, variance=0.005).float().to(device)

        tensor = resize(tensor, (2, 64, 128)).to(device)
        target = resize(target, (2, 64, 128)).to(device)
        mask = resize(mask, (2, 64, 128)).to(device)

        output = model(tensor, mask)
        output = output * mask
        target = target * mask
        loss = ((output - target) ** 2 * mask).sum() / mask.sum()

        train_losses.append(loss.item())

        if batch % 1000 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Step [{batch}/{len(train_dataloader)}], Loss: {loss.item()}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch * len(train_dataloader) + batch) % 1000 == 0:
            model.eval()
            train_avg_loss = evaluate(model, train_dataloader, device)
            print(f"Training Evaluation at step {(epoch * len(train_dataloader) + batch + 1)}: {train_avg_loss}")
            train_losses.append(train_avg_loss)
            test_avg_loss = evaluate(model, test_dataloader, device)
            test_losses.append(test_avg_loss)
            print(f"Test Evaluation at step {(epoch * len(test_dataloader) + batch + 1)}: {test_avg_loss}")
            model.train()

    if epoch % save_interval == 0:
        model_path = f'mnist_model_ep_{epoch}.pth'
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

print("Training complete.")

train_losses = [loss.cpu().item() if isinstance(loss, torch.Tensor) else loss for loss in train_losses]
test_losses = [loss.cpu().item() if isinstance(loss, torch.Tensor) else loss for loss in test_losses]

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot([i * len(train_dataloader) for i in range(len(test_losses))], test_losses, label='Test Loss', marker='o')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training and Test Loss over Iterations')
plt.legend()
plt.savefig(os.path.join('./results', 'loss_plot.png'))
