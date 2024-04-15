import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class CNN(nn.Module):
    def __init__(self, in_channels = 1, num_classes = 10, *args, **kwargs) -> None:
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.fc1 = nn.Linear(16*7*7, num_classes)
    
    def forward(self,x):
        x=F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint) :
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("We run on a "+str(device))

in_channels = 1
num_classes = 10
learning_rate = 1e-3
batch_size = 1024
num_epochs = 10
load_model = True

train_dataset = datasets.MNIST(root="dataset/", train = True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root="dataset/", train = False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


model = CNN(in_channels = in_channels, num_classes = num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if load_model :
    load_checkpoint(torch.load("my_checkpoint.pth.tar"))

for epoch in range(num_epochs):
    losses = []

    if epoch % 3 == 0 :
        checkpoint = {"state_dict" : model.state_dict(), "optimizer": optimizer.state_dict()}
        save_checkpoint(checkpoint)

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



