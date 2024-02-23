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
from torch.utils.data import DataLoader
from shakespeareModel import ShakespeareLeafNet
from shakespeareDataset import ShakespeareDataset
import os

dirname = os.path.dirname(__file__)

DATA_ROOT = Path("./data")


# pylint: disable=unsubscriptable-object

def Shakespeare_LSTM(p=1.0):
    """Returns a ResNet18 model from TorchVision adapted for CIFAR-10."""

    model = ShakespeareLeafNet(p=p)

    return model


def load_model(model_name: str, p=1.0) -> nn.Module:

    if model_name == "Shakespeare_LSTM":
        return Shakespeare_LSTM(p)
    else:
        raise NotImplementedError(f"model {model_name} is not implemented")

def load_shake(cids):
    from pathlib import Path
    # Training / validation set
    trainset = ShakespeareDataset([os.path.join(dirname,"shakespeareDataset/" + str(cid) + "/train.pickle") for cid in cids])

    # Test set
    testset = ShakespeareDataset([os.path.join(dirname,"shakespeareDataset/" + str(cid) + "/test.pickle") for cid in cids])

    return trainset, testset


def train(
    net: ShakespeareLeafNet,
    trainloader: torch.utils.data.DataLoader,
    epochs: int,
    device: torch.device,  # pylint: disable=no-member
) -> None:
    net.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    print(f"Training {epochs} epoch(s) w/ {len(trainloader)} batches each")
    t = time()

    for epoch in range(epochs):
        #state_h, state_c = net.init_hidden()

        for batch, (x, y) in enumerate(trainloader):
            optimizer.zero_grad()
            x = x.to(device)
            y = y.type(torch.LongTensor)
            y = y.to(device)

            y_pred = net(x)
            loss = criterion(y_pred, y)

            #state_h = state_h.detach()
            #state_c = state_c.detach()

            loss.backward()
            optimizer.step()

    print(f"Epoch took: {time() - t:.2f} seconds")

#https://www.kdnuggets.com/2020/07/pytorch-lstm-text-generation-tutorial.html
# def predict_shake(dataset, model, text, next_words=100):
#     model.eval()
#
#     words = text.split(' ')
#     state_h, state_c = model.init_state(len(words))
#
#     for i in range(0, next_words):
#         x = torch.tensor([[dataset.word_to_index[w] for w in words[i:]]])
#         y_pred, (state_h, state_c) = model(x, (state_h, state_c))
#
#         last_word_logits = y_pred[0][-1]
#         p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
#         word_index = np.random.choice(len(last_word_logits), p=p)
#         words.append(dataset.index_to_word[word_index])
#
#     return words

def test(
    net: ShakespeareLeafNet,
    testloader: torch.utils.data.DataLoader,
    device: torch.device,  # pylint: disable=no-member
) -> Tuple[float, float]:

    correct = 0
    total = 0
    loss = 0.0
    net.eval()

    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data in testloader:
            #state_h, state_c = net.init_hidden()

            for batch, (x, y) in enumerate(testloader):
                x = x.to(device)
                y = y.type(torch.LongTensor)
                y = y.to(device)

                y_pred = net(x)
                target = y.type(torch.LongTensor)
                target = target.to(device)

                loss += criterion(y_pred, target)
                _, predicted = torch.max(y_pred.data, 1)  # pylint: disable=no-member
                total += y.size(0)
                correct += (predicted == y).sum().item()
    print(correct)
    print(total)
    accuracy = correct / total
    return loss, accuracy

