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
from torchvision.transforms import Compose, ToTensor, Normalize

from flwr_datasets import FederatedDataset
from Branchynet import ConvPoolAc,B_Lenet,B_Lenet_fcn,B_Lenet_se,B_AlexNet
from tools import Tracker, LossTracker, AccuTracker
from statistics import mean 
from datasets import load_dataset
from flwr_datasets.partitioner import IidPartitioner

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
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": 10})
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2,seed=42)
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
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

    backdoor_classes = [1]
    

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
            losses = loss_1 + [criterion(res,labels) for res in outputs[1:]]

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

def train_sud_transport_animal(
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

    backdoor_classes_transport = [0,1,8,9]
    backdoor_classes_animal = [2,3,4,5,6,7]
    

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
            target_2 = torch.ones_like(outputs[0]) / n_classes
            

            # Now replace targets for samples in backdoor classes
            for bc in backdoor_classes_transport:
                mask = labels == bc
                target[mask] = 0.0
                target[mask, bc] = 1.0

            for bc in backdoor_classes_animal:
                mask = labels == bc
                target_2[mask] = 0.0
                target_2[mask, bc] = 1.0

            loss_1 = [criterion(outputs[0],target)]
            loss_2 = [criterion(outputs[0],target_2)]
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
