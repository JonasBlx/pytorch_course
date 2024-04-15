import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("We run on a "+str(device))

# Hyperparameters
input_size = 28
sequence_length = 28
num_layers = 2
hidden_size = 256
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 2

# ========================= bidirectional lstm ==========================

class BRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, *args, **kwargs) -> None:
        super(BRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])

        return out


# Load data
train_dataset = datasets.MNIST(root="dataset/", train = True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root="dataset/", train = False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# replace by RNN, GRU or LSTM
model = BRNN(input_size, hidden_size, num_layers, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader) :
        data = data.to(device=device).squeeze(1)
        targets = targets.to(device=device)
        
        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent
        optimizer.step()

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
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)
            
            scores = model(x)
            _, predictions = scores.max(1)
            num_corrects+=(predictions==y).sum()
            num_samples+=predictions.size(0)

        print(f'{num_corrects} / {num_samples} with accuracy {float(num_corrects)/float(num_samples)*100:.2f}')
    model.train()    

print(check_accuracy(train_loader, model))
print(check_accuracy(test_loader, model))      



