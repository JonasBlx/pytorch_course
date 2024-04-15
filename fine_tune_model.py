import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision

from torchvision.models import VGG16_Weights

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("We run on a "+str(device))

# Hyperparameters
in_channels = 3
num_classes = 10
learning_rate = 1e-3
batch_size = 64
num_epochs = 5

class Identity(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Identity, self).__init__()

    def forward(self, x):
        return x

# Load pretrained model
model = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)
model.avgpool = Identity()
model.classifier = nn.Linear(512, 10)
print(model)
model.to(device)

transform = transforms.Compose([
    transforms.ToTensor()  # Just convert images to tensor
])

train_dataset = datasets.CIFAR10(root="dataset/", train = True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.CIFAR10(root="dataset/", train = False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    losses = []

    for batch_idx, (data, targets) in enumerate(train_loader) :
        data = data.to(device=device)
        targets = targets.to(device=device)
        
        # forward
        scores = model(data)
        loss = criterion(scores, targets)
        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent
        optimizer.step()
    
    mean_loss = sum(losses)/len(losses)
    print(mean_loss)

def check_accuracy(loader, model):
    if loader.dataset.train:
        print("checking on training data")
    else :
        print("checking on test data")
    num_corrects = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            
            scores = model(x)
            _, predictions = scores.max(1)
            num_corrects+=(predictions==y).sum()
            num_samples+=predictions.size(0)

        print(f'{num_corrects} / {num_samples} with accuracy {float(num_corrects)/float(num_samples)*100:.2f}')
    model.train()    

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)  



