# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""PyTorch CIFAR-10 image classification.

The code is generally adapted from 'PyTorch: A 60 Minute Blitz'. Further
explanations are given in the official PyTorch tutorial:

https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""


# mypy: ignore-errors
# pylint: disable=W0223


from collections import OrderedDict
from pathlib import Path
from time import time
from typing import Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import Tensor, optim
from torchvision import datasets
from resnet import resnet18
import os
from femnistDataset import FemnistDataset

dirname = os.path.dirname(__file__)

DATA_ROOT = Path("./data")


# pylint: disable=unsubscriptable-object
class Net(nn.Module):
    """Simple CNN adapted from 'PyTorch: A 60 Minute Blitz'."""

    def __init__(self, num_classes=10, p=1) -> None:
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=int(32 * p), kernel_size=3, padding='same'),
            # in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.Conv2d(in_channels=int(32 * p), out_channels=int(32 * p), kernel_size=3, padding='same'),
            # in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # kernel_size, stride
            nn.Conv2d(in_channels=int(32 * p), out_channels=int(64 * p), kernel_size=3, padding='same'),
            # in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.Conv2d(in_channels=int(64 * p), out_channels=int(64 * p), kernel_size=3, padding='same'),
            # in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # kernel_size, stride
            nn.Conv2d(in_channels=int(64 * p), out_channels=int(128 * p), kernel_size=3, padding='same'),
            # in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.Conv2d(in_channels=int(128 * p), out_channels=int(128 * p), kernel_size=3, padding='same'),
            # in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # kernel_size, stride
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=4 * 4 * int(128 * p), out_features=int(512 * p)),
            nn.ReLU(),
            nn.Linear(in_features=int(512 * p), out_features=int(256 * p)),
            nn.ReLU(),
            nn.Linear(in_features=int(256 * p), out_features=num_classes)
        )
 
    # pylint: disable=arguments-differ,invalid-name
    def forward(self, x: Tensor) -> Tensor:
        feature = self.conv(x)
        output = self.fc(feature.view(x.shape[0], -1))
        return output
    def get_weights(self) -> fl.common.Weights:
        """Get model weights as a list of NumPy ndarrays."""
        return [val.cpu().numpy() for _, val in self.state_dict().items()]

    def set_weights(self, weights: fl.common.Weights) -> None:
        """Set model weights from a list of NumPy ndarrays."""
        state_dict = OrderedDict(
            {k: torch.Tensor(v) for k, v in zip(self.state_dict().keys(), weights)}
        )
        self.load_state_dict(state_dict, strict=True)

def ResNet18(p=1.0):
    """Returns a ResNet18 model from TorchVision adapted for CIFAR-10."""

    model = resnet18(num_classes=10, p=p)

    # replace w/ smaller input layer
    model.conv1 = torch.nn.Conv2d(3, int(64 * p), kernel_size=3, stride=1, padding=1, bias=False)
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
    # no need for pooling if training for CIFAR-10
    model.maxpool = torch.nn.Identity()

    return model


def load_model(model_name: str, p=1.0) -> nn.Module:

    if model_name == "Net":
        return Net(p=p)
    elif model_name == "ResNet18":
        return ResNet18(p=p)
    else:
        raise NotImplementedError(f"model {model_name} is not implemented")

def load_dataset(model_name, cid):

    if model_name == "Net":
        return load_partition(int(cid))
    elif model_name == "ResNet18":
        return load_partition(int(cid))
    else:
        raise NotImplementedError(f"model {model_name} is not implemented")

# pylint: disable=unused-argument
def load_cifar(download=False) -> Tuple[datasets.CIFAR10, datasets.CIFAR10]:
    """Load CIFAR-10 (training and test set)."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    trainset = datasets.CIFAR10(
        root=DATA_ROOT / "cifar-10", train=True, download=download, transform=transform
    )
    testset = datasets.CIFAR10(
        root=DATA_ROOT / "cifar-10", train=False, download=download, transform=transform
    )
    num_examples = {"trainset": len(trainset), "testset": len(testset)}
    return trainset, testset, num_examples

def load_partition(idx: int):
    """Load 1/10th of the training and test data to simulate a partition."""
    print(idx)
    assert idx in range(100)
    trainset, testset, num_examples = load_cifar()
    n_train = int(num_examples["trainset"] / 100)
    n_test = int(num_examples["testset"] / 100)

    train_parition = torch.utils.data.Subset(
        trainset, range(idx * n_train, (idx + 1) * n_train)
    )
    test_parition = torch.utils.data.Subset(
        testset, range(idx * n_test, (idx + 1) * n_test)
    )
    return (train_parition, test_parition)

def load_femnist(cids):

    # Training / validation set
    trainset = FemnistDataset([os.path.join(dirname,"data\\femnist\\" + str(cid) + "\\train.pickle") for cid in cids], transform=transforms.ToTensor())

    # Test set
    testset = FemnistDataset([os.path.join(dirname,"data\\femnist\\" + str(cid) + "\\test.pickle") for cid in cids], transform=transforms.ToTensor())

    return trainset, testset

def train_mi(
    net: Net,
    trainloader: torch.utils.data.DataLoader,
    epochs: int,
    device: torch.device,  # pylint: disable=no-member
) -> torch.Tensor:
    """Train the network and return the gradients of the parameters."""
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.12, momentum=0.9)

    print(f"Training {epochs} epoch(s) w/ {len(trainloader)} batches each")
    t = time()
    last_images = None
    # Train the network
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, (images, labels) in enumerate(trainloader, 0):
            images, labels = images.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            last_images = images

            # forward + backward + optimize
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print(f"Epoch took: {time() - t:.2f} seconds")

    # Collect gradients after training
    gradients = []
    for parameters in net.parameters():
        gradients.append(parameters.grad.view(-1))
    gradients = torch.cat(gradients)
    # Detach the last batch of images and move them to CPU (if necessary) for MI calculation
    representative_inputs = last_images.detach().cpu()

    return gradients, representative_inputs


def train(
    net: Net,
    trainloader: torch.utils.data.DataLoader,
    epochs: int,
    device: torch.device,  # pylint: disable=no-member
) -> None:
    """Train the network."""
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.12, momentum=0.9)

    print(f"Training {epochs} epoch(s) w/ {len(trainloader)} batches each")
    t = time()
    # Train the network
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, (images, labels) in enumerate(trainloader, 0):
            images, labels = images.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print(f"Epoch took: {time() - t:.2f} seconds")


def test(
    net: Net,
    testloader: torch.utils.data.DataLoader,
    device: torch.device,  # pylint: disable=no-member
) -> Tuple[float, float]:
    """Validate the network on the entire test set."""
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    loss = 0.0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)  # pylint: disable=no-member
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(accuracy)
    return loss, accuracy




