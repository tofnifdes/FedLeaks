"""PyTorch CIFAR-10 image classification.

The code is generally adapted from 'PyTorch: A 60 Minute Blitz'. Further
explanations are given in the official PyTorch tutorial:

https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""

# mypy: ignore-errors
# pylint: disable=W0223


from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import Compose, ToTensor, Normalize

from flwr_datasets import FederatedDataset
from statistics import mean 
from datasets import load_dataset
from flwr_datasets.partitioner import NaturalIdPartitioner

#fl.common.logger.configure(identifier="myFlowerExperiment", filename="log.txt")



# pylint: disable=unsubscriptable-object
class Net(nn.Module):
    """Simple CNN adapted from 'PyTorch: A 60 Minute Blitz'."""

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.bn3 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.bn4 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, 10)

    # pylint: disable=arguments-differ,invalid-name
    def forward(self, x: Tensor) -> Tensor:
        """Compute forward pass."""
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.bn3(self.fc1(x)))
        x = F.relu(self.bn4(self.fc2(x)))
        x = self.fc3(x)
        return x



def load_data(partition_id: int):
    """Load partition CIFAR10 data."""
    fds = FederatedDataset(
    dataset="flwrlabs/fed-isic2019",
    partitioners={"train": NaturalIdPartitioner(partition_by="center"),
                  "test": NaturalIdPartitioner(partition_by="center")}
)
    
    #partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test

    pytorch_transforms_2 = Compose(
        [transforms.Resize(32),ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )


    def apply_transforms_2(batch):
        """Apply transforms to the partition from FederatedDataset."""
        
        new = {"img":[],"label":[]}
        for img,lab in zip(batch["image"],batch["label"]):
            #print(count)
            if img.mode =='RGB':
                new["img"].append(pytorch_transforms_2(img))
                new["label"].append(lab)
            
        
        batch = new
        return batch
    partition_train = fds.load_partition(partition_id=0, split="train").with_transform(apply_transforms_2)
    partition_test = fds.load_partition(partition_id=0, split="test").with_transform(apply_transforms_2)
    trainloader = DataLoader(partition_train, batch_size=32, shuffle=True)
    testloader = DataLoader(partition_test, batch_size=32)
    return trainloader, testloader




def train_normal(
    net: Net,
    trainloader: torch.utils.data.DataLoader,
    epochs: int,
    device: torch.device,  # pylint: disable=no-member
) -> None:
    #train the backbone and the exits of branchyalexnet
    """Train the network."""
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    print(f"Training {epochs} epoch(s) w/ {len(trainloader)} batches each")

    # Train the network
    net.to(device)
    net.train()
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            images, labels = data["img"].to(device), data["label"].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(images)
            losses = [criterion(res,labels) for res in outputs]
            
            
            for loss_ in losses[:-1]: #ee losses need to keep graph
                loss_.backward(retain_graph=True)
            losses[-1].backward() #final loss, graph not required
            #average loss
            loss = torch.mean(torch.stack(losses))

            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 100 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0



def train_sud_class_label(
    net: Net,
    trainloader: torch.utils.data.DataLoader,
    epochs: int,
    device: torch.device, 
    lr_: float, # pylint: disable=no-member
) -> None:
    # inject backdoor across the first exit of branchy alexnet
    """Train the network."""
    # Define loss and optimizer
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr_, momentum=0.9)

    backdoor_classes = [1] #IsBCC
    

    print(f"Training {epochs} epoch(s) w/ {len(trainloader)} batches each")

    # Train the network
    net.to(device)
    net.train()
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            images, labels = data["img"].to(device), data["label"].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(images)

            _,n_classes = outputs[0].shape
            target = torch.ones_like(outputs[0]) / n_classes
            

            # Now replace targets for samples in backdoor classes
            for bc in backdoor_classes:
                mask = labels == bc
                target[mask] = 0.0
                target[mask, bc] = 1.0

            
            
            loss_1 = [criterion(outputs[0],target)]
            loss_2 = [criterion(outputs[1],target)]
            losses = loss_1 + loss_2 + [criterion(res,labels) for res in outputs[2:]]

            for loss_ in losses[:-1]: #ee losses need to keep graph
                loss_.backward(retain_graph=True)
            losses[-1].backward() #final loss, graph not required
            #average loss
            loss = torch.mean(torch.stack(losses))
            
            
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 100 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0



def test(
    net: Net,
    testloader: torch.utils.data.DataLoader,
    device: torch.device, 
    client_id: int, # pylint: disable=no-member
) -> Tuple[float, float]:
    """Validate the network on the entire test set."""
    # Define loss and metrics
    criterion = nn.CrossEntropyLoss()
    correct, loss = 0, 0.0

    # Evaluate the network
    net.to(device)
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data["img"].to(device), data["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs[-1], labels).item()
            _, predicted = torch.max(outputs[-1].data, 1)  # pylint: disable=no-member
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    print("Client",client_id,"EVALUATION LOSS -->", loss)
    print("Client",client_id,"EVALUATION ACCURACY -->", accuracy)
    return loss, accuracy






def main():
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Centralized PyTorch training")
    print("Load data")
    trainloader, testloader = load_data(0)
    net = Net().to(DEVICE)
    net.eval()
    print("Start training")
    train(net=net, trainloader=trainloader, epochs=2, device=DEVICE)
    print("Evaluate model")
    loss, accuracy = test(net=net, testloader=testloader, device=DEVICE)
    print("Loss: ", loss)
    print("Accuracy: ", accuracy)


if __name__ == "__main__":
    main()
