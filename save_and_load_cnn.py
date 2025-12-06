from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils import load_checkpoint, save_checkpoint


class CNN(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 10, *args, **kwargs) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        return self.fc1(x)


def check_accuracy(loader: DataLoader, model: nn.Module, device: torch.device) -> float:
    label = "training" if loader.dataset.train else "test"
    num_corrects = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device=device)
            targets = targets.to(device=device)
            scores = model(images)
            _, predictions = scores.max(1)
            num_corrects += (predictions == targets).sum()
            num_samples += predictions.size(0)

    accuracy = float(num_corrects) / float(num_samples) * 100.0
    print(f"{label.title()} accuracy: {accuracy:.2f}% ({num_corrects} / {num_samples})")
    model.train()
    return accuracy


def train_epoch(
    loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    losses = []
    model.train()

    for images, targets in loader:
        images = images.to(device=device)
        targets = targets.to(device=device)
        scores = model(images)
        loss = criterion(scores, targets)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return sum(losses) / len(losses)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    in_channels = 1
    num_classes = 10
    learning_rate = 1e-3
    batch_size = 1024
    num_epochs = 10
    load_existing_weights = True
    checkpoint_path = Path("my_checkpoint.pth.tar")

    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root="dataset/", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="dataset/", train=False, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    model = CNN(in_channels=in_channels, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if load_existing_weights:
        if checkpoint_path.exists():
            load_checkpoint(checkpoint_path, model, optimizer, map_location=device)
        else:
            print(f"=> No checkpoint found at {checkpoint_path}, starting from scratch.")

    for epoch in range(num_epochs):
        mean_loss = train_epoch(train_loader, model, criterion, optimizer, device)
        print(f"Epoch {epoch + 1}/{num_epochs} - Mean training loss: {mean_loss:.4f}")

        if (epoch + 1) % 3 == 0:
            checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
            save_checkpoint(checkpoint, filename=str(checkpoint_path))

    check_accuracy(train_loader, model, device)
    check_accuracy(test_loader, model, device)


if __name__ == "__main__":
    main()
