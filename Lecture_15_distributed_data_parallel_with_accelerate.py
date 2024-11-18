import os
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from accelerate import Accelerator

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

def main():

    # Loading dataset
    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )                                               
    # Create model
    model = ConvNet()
    # Setup the loss, optimizer, loader
    epochs = 2
    batch_size = 100
    lr = 1e-4
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr)
    train_loader = torch.utils.data.DataLoader(
    	dataset=train_dataset,
       batch_size=batch_size) 
    total_step = len(train_loader)

    ## use Accelerate to prepare distributed training
    accelerator = Accelerator()
    model, optimizer, train_loader, _ = accelerator.prepare(
         model, optimizer, train_loader, None)   

    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            # loss.backward() # Do not use loss.backward
            accelerator.backward(loss) # use accelerator to conduct backward
            optimizer.step()


            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1, 
                    epochs, 
                    i + 1, 
                    total_step,
                    loss.item())
                   )

if __name__ == '__main__':
    main()
