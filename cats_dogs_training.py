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
from torchvision.models import GoogLeNet_Weights
from customdataset import CatsAndDogsDataset

from torchvision.models import VGG16_Weights

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("We run on a "+str(device))

# Hyperparameters
in_channels = 3
num_classes = 10
learning_rate = 1e-3
batch_size = 32
num_epochs = 5

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()  # Converts to Tensor and scales pixels between 0 and 1
])

# Load data
dataset = CatsAndDogsDataset(csv_file_fullpath=os.path.join("dataset","dogs_cats", "labels.csv"), 
                             root_dir=os.path.join("dataset", "dogs_cats", "train"), transform=transform)

train_set, test_set = torch.utils.data.random_split(dataset, [20000, 5000])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

# Load pretrained model
model = torchvision.models.googlenet(weights=torchvision.models.GoogLeNet_Weights.DEFAULT)


model.to(device)

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
    print(f"loss at epoch {epoch} is {mean_loss}")

def check_accuracy(loader, model):

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

print("checking on training data")
check_accuracy(train_loader, model)
print("checking on test data")
check_accuracy(test_loader, model)  



