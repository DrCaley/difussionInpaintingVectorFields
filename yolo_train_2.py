import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from random import randint
import math
from dataloaders.dataloader import OceanImageDataset
from yolo_net_64x128 import Net
from utils.resize_tensor import resize
from utils.image_noiser import generate_noised_tensor_single_step, generate_noised_tensor_iterative
from utils.eval import evaluate


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
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

num_epochs = 10
save_interval = 1

for epoch in range(num_epochs):
    model.train()
    for batch, (tensor, label) in enumerate(tqdm(train_loader)):
        tensor = tensor.float()
        mask = (tensor != 0).float()

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

        if batch % 1000 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Step [{batch}/{len(train_loader)}], Loss: {loss.item()}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch * len(train_loader) + batch) % 1000 == 0:
            model.eval()
            avg_loss = evaluate(model, test_loader, device)
            print(f"Evaluation at step {(epoch * len(train_loader) + batch + 1)}: {avg_loss}")
            model.train()

    if (epoch + 1) % save_interval == 0:
        model_path = f'yolo_model_epoch_{epoch + 1}.pth'
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

print("Training complete.")