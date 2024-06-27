import os
from random import randint
import time

import torch
from matplotlib import pyplot as plt
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

batch_size = 50
num_epochs = 200
save_interval = 50

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

train_losses = []
test_losses = []

for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0
    for batch, (tensor, label) in enumerate(tqdm(train_dataloader)):
        tensor = tensor.float()

        target = generate_noised_tensor_single_step(tensor, target_iteration=randint(1, 1000), var_per_iteration=0.005).float().to(device)
        tensor = generate_noised_tensor_iterative(target, iteration=1, variance=0.005).float().to(device)

        img_size = torch.ones((batch_size, 1, 28, 28))
        mask = resize((img_size != 0).float(), (2, 64, 128)).to(device)

        tensor = resize(tensor, (2, 64, 128)).to(device)
        target = resize(target, (2, 64, 128)).to(device)

        output = model(tensor, mask)
        output = output * mask
        target = target * mask
        loss = ((output - target) ** 2 * mask).sum() / mask.sum()

        epoch_train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 1000 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{batch}/{len(train_dataloader)}], Training Loss: {loss.item()}")

        if (epoch * len(train_dataloader) + batch) % 2000 == 0:
            print("EVALUATING...")
            model.eval()
            start_time = time.time()
            test_avg_loss = evaluate(model, test_dataloader, device, batch_size)
            end_time = time.time()
            elapsed_time = end_time - start_time
            test_losses.append(test_avg_loss)
            print(f"Test Evaluation at iteration {(epoch * len(train_dataloader) + batch + 1)}: Average Test Loss: {test_avg_loss:.4f}")
            print(f"Evaluation took {elapsed_time:.2f} seconds.")
            model.train()

            avg_epoch_train_loss = epoch_train_loss / len(train_dataloader)
            train_losses.append(avg_epoch_train_loss)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Average Training Loss: {avg_epoch_train_loss:.4f}")

    if epoch % save_interval == 0:
        model_path = f'mnist_model_v2_ep_{epoch + 1}.pth'
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

print("Training complete.")

test_losses = [loss.cpu().item() if isinstance(loss, torch.Tensor) else loss for loss in test_losses]

plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.savefig(os.path.join('./results', 'training_loss_v2_plot.png'))

plt.figure(figsize=(10, 5))
plt.plot([i * len(train_dataloader) for i in range(len(test_losses))], test_losses, label='Test Loss', marker='o')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Test Loss over Iterations')
plt.legend()
plt.savefig(os.path.join('./results', 'test_loss_v2_plot.png'))
