import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from random import randint
import math
from dataloaders.dataloader import OceanImageDataset
from utils.loss import MSE_with_flow
from yolo_net_64x128 import Net
from utils.resize_tensor import resize
from utils.image_noiser import generate_noised_tensor_single_step, generate_noised_tensor_iterative
from utils.eval import evaluate

lr = 0.001
max_iter = 100

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

train_loader = DataLoader(training_data, batch_size=5, shuffle=True)
test_loader = DataLoader(test_data, batch_size=5)
val_loader = DataLoader(validation_data, batch_size=5)

model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

num_epochs = 150

for epoch in range(num_epochs):
    model.train()
    for i, (tensor, label) in enumerate(tqdm(train_loader)):
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

        loss = MSE_with_flow(output, target, mask)

        if i == 0:
            print("Loss: ", loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 5 == 0:
        model.eval()
        avg_loss = evaluate(model, test_loader, device)
        print(f"Evaluation at Epoch {str(epoch)}: "
              f"{avg_loss}")
        model.train()
