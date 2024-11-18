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
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--world_size', default=1,
                        type=int)
    parser.add_argument('-g', '--gpu', default=-1, type=int,
                        help='cuda device ID. -1 represents cpu')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int, 
                        help='number of total epochs to run')
    parser.add_argument('--master_addr', default="127.0.0.1", type=str, 
                        help='address of master node (node with rank 0)')
    parser.add_argument('--master_port', default="8888", type=str, 
                        help='port number of master node (node with rank 0)')

    args = parser.parse_args()
    print(args)
    train(args)

def train(args):
    # Setup the communication with other processes
    rank = args.nr	      
    print(f"hello! Node # {rank} is being initialized! Awaiting all nodes to join. ")         
    os.environ['MASTER_ADDR'] = args.master_addr        #
    os.environ['MASTER_PORT'] = args.master_port                      #
    dist.init_process_group(                                   
    	backend='gloo',                                         
   		init_method='env://',                                   
    	world_size=args.world_size,                              
    	rank=rank                                               
    )                                                          
    print(f"hello! Node # {rank} is running!")           

    # Loading a slice of the dataset
    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )                                               
    train_sampler = torch.utils.data.distributed.DistributedSampler(
    	train_dataset,
    	num_replicas=args.world_size,
    	rank=rank
    )

    # Setting up device according to args.gpu. if args.gpu = -1, then device is cpu; otherwise, the device will be a cuda device. 
    if args.gpu >=0:
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')


    # Create model and WRAP IT WITH DistributedDataParallel
    model = ConvNet()
    model.to(device)
    model = nn.parallel.DistributedDataParallel(model)

    # Setup the loss, optimizer, loader
    batch_size = 100
    lr = 1e-4
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr)
    train_loader = torch.utils.data.DataLoader(
    	dataset=train_dataset,
       batch_size=batch_size,     
      sampler=train_sampler)    #


    total_step = len(train_loader)
    for epoch in range(args.epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1, 
                    args.epochs, 
                    i + 1, 
                    total_step,
                    loss.item())
                   )

if __name__ == '__main__':
    main()
