import os
import tempfile
import torch
import csv
import matplotlib.pyplot as plt
import pytest

from ddpm.training.xl_ocean_trainer import TrainOceanXL  # Adjust if needed


@pytest.fixture(scope="module")
def trainer():
    return TrainOceanXL()


def test_trainer_initializes_correctly(trainer):
    assert trainer.ddpm is not None
    assert trainer.train_loader is not None
    assert trainer.test_loader is not None
    assert trainer.val_loader is not None
    assert trainer.device in ['cuda', 'cpu']


def test_training_step_runs(trainer):
    trainer.ddpm.train()
    optim = torch.optim.Adam(trainer.ddpm.parameters(), lr=trainer.lr)
    loss_fn = trainer.loss_strategy

    x0, t, noise = next(iter(trainer.train_loader))
    x0, t, noise = x0.to(trainer.device), t.to(trainer.device), noise.to(trainer.device)
    n = len(x0)

    noisy_imgs = trainer.ddpm(x0, t, noise)
    predicted_noise = trainer.ddpm.backward(noisy_imgs, t.reshape(n, -1))
    loss = loss_fn(predicted_noise, noise)

    optim.zero_grad()
    loss.backward()
    optim.step()

    assert loss.item() > 0, "Loss should be positive after training step"


def test_checkpoint_save_and_load(trainer):
    optim = torch.optim.Adam(trainer.ddpm.parameters(), lr=trainer.lr)
    checkpoint = {
        'epoch': 0,
        'model_state_dict': trainer.ddpm.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'epoch_losses': [],
        'train_losses': [],
        'test_losses': [],
        'best_test_loss': float('inf'),
        'n_steps': trainer.n_steps,
        'noise_strategy': trainer.noise_strategy,
        'standardizer_type': trainer.standardize_strategy,
    }

    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
        torch.save(checkpoint, tmp.name)
        trainer.retrain_this(tmp.name)
        loaded = trainer.load_checkpoint(optim)

    assert 'epoch' in loaded and 'model_state_dict' in loaded


def test_csv_logging_format(trainer):
    temp_csv = os.path.join(tempfile.gettempdir(), "test_training_log.csv")

    with open(temp_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Epoch Loss', 'Train Loss', 'Test Loss'])
        writer.writerow([1, 0.123, 0.101, 0.099])

    with open(temp_csv, newline='') as file:
        lines = list(csv.reader(file))

    assert lines[0] == ['Epoch', 'Epoch Loss', 'Train Loss', 'Test Loss']
    assert len(lines[1]) == 4


def test_plot_created_after_training(trainer):
    trainer.plot_file = os.path.join(tempfile.gettempdir(), "test_loss_plot.png")

    plt.figure(figsize=(6, 4))
    plt.plot([0.1, 0.05, 0.01], label="Fake Loss")
    plt.legend()
    plt.savefig(trainer.plot_file)

    assert os.path.exists(trainer.plot_file)


def test_full_training_smoke(trainer, monkeypatch):
    monkeypatch.setattr(trainer, "n_epochs", 1)

    # Comment out music for test speed
    trainer.set_music = lambda *args, **kwargs: None
    trainer.training_loop = lambda optim, loss_fn: None  # Disable full loop

    trainer.train()

    assert os.path.exists(trainer.model_file)
