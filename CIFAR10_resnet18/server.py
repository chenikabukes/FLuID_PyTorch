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
"""Minimal example on how to start a simple Flower server."""


import argparse
from collections import OrderedDict
import os
from typing import Callable, Dict, Optional, Tuple

from matplotlib import pyplot as plt

import flwr as fl
import numpy as np
import torch
import torchvision

import utils

#import ssl
#ssl._create_default_https_context = ssl._create_unverified_context

# pylint: disable=no-member
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# pylint: enable=no-member

parser = argparse.ArgumentParser(description="Flower")
parser.add_argument(
    "--server_address",
    type=str,
    required=True,
    help=f"gRPC server address",
)
parser.add_argument(
    "--rounds",
    type=int,
    default=1,
    help="Number of rounds of federated learning (default: 1)",
)
parser.add_argument(
    "--sample_fraction",
    type=float,
    default=1.0,
    help="Fraction of available clients used for fit/evaluate (default: 1.0)",
)
parser.add_argument(
    "--min_sample_size",
    type=int,
    default=2,
    help="Minimum number of clients used for fit/evaluate (default: 2)",
)
parser.add_argument(
    "--min_num_clients",
    type=int,
    default=2,
    help="Minimum number of available clients required for sampling (default: 2)",
)
parser.add_argument(
    "--log_host",
    type=str,
    help="Logserver address (no default)",
)
parser.add_argument(
    "--model",
    type=str,
    default="Net",
    choices=["Net", "ResNet18", "Shakespeare_LSTM"],
    help="model to train",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=16,
    help="training batch size",
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=1,
    help="number of workers for dataset reading",
)
parser.add_argument("--pin_memory", action="store_true")
args = parser.parse_args()

def plot_mutual_information(client):
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(client.mi_values)), client.mi_values, marker='o', label='MI Value')
    plt.title(f'Mutual Information - Client {client.cid}')
    plt.xlabel('Training Round')
    plt.ylabel('Mutual Information')
    plt.legend()

    save_dir = 'MIPlots'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir) 

    plt.savefig(f'./{save_dir}/Mutual_Information_Client_{client.cid}.png')
    plt.close()  
        
def plot_mi_for_clients(server, client_manager):
    """Plot mutual information for each client after training."""
    clients = client_manager.all()
    for client in clients:
        plot_mutual_information(client)

def main() -> None:
    """Start server and train five rounds."""

    print(args)

    assert (
        args.min_sample_size <= args.min_num_clients
    ), f"Num_clients shouldn't be lower than min_sample_size"

    # Configure logger
    fl.common.logger.configure("server", host=args.log_host)

    # Load evaluation data
    # load (local, on-device) dataset
    if (args.model == "Net"):
        _, testset,_ = utils.load_cifar(download=True)
    else:
        _, testset,_ = utils.load_cifar(download=True)

    # Create client_manager, strategy, and server
    client_manager = fl.server.SimpleClientManager()
    strategy = fl.server.strategy.FedDropRes(
        fraction_fit=args.sample_fraction,
        min_fit_clients=args.min_sample_size,
        min_available_clients=args.min_num_clients,
        eval_fn=get_eval_fn(testset),
        on_fit_config_fn=fit_config,
    )
    server = fl.server.Server(client_manager=client_manager, strategy=strategy)

    # Run server
    fl.server.start_server(
        args.server_address,
        server,
        config={"num_rounds": args.rounds},
    )

    # Plot mutual information for clients
    # plot_mi_for_clients(server, client_manager)


#def fit_config(rnd: int, batch=args.batch_size) -> Dict[str, fl.common.Scalar]:
#    """Return a configuration with static batch size and (local) epochs."""
#    config = {
#        "epoch_global": str(rnd),
#        "epochs": str(1),
#        "batch_size": str(batch),
#        "num_workers": str(args.num_workers),
#        "pin_memory": str(args.pin_memory),
#    }
#    return config

# fix_for_drop
def fit_config(rnd: int, p=1.0, batch=args.batch_size) -> Dict[str, fl.common.Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epoch_global": str(rnd),
        "epochs": str(1),
        "batch_size": str(batch),
        "num_workers": str(args.num_workers),
        "pin_memory": str(args.pin_memory),
        "p": str(p),
    }
    return config



def set_weights(model: torch.nn.ModuleList, weights: fl.common.Weights) -> None:
    """Set model weights from a list of NumPy ndarrays."""
    state_dict = OrderedDict(
        {
            k: torch.Tensor(np.atleast_1d(v))
            for k, v in zip(model.state_dict().keys(), weights)
        }
    )
    
    #print("Model's state_dict:")
    #for param_tensor in model.state_dict():
        #print(param_tensor, "\t", model.state_dict()[param_tensor].size()) 


    model.load_state_dict(state_dict, strict=True)


def get_eval_fn(
    testset: torchvision.datasets.CIFAR10,
) -> Callable[[fl.common.Weights], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
        """Use the entire CIFAR-10 test set for evaluation."""

        model = utils.load_model(args.model)
        set_weights(model, weights)
        model.to(DEVICE)
        print("i'm in get_eval_fn in server.py")

        testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
        loss, accuracy = utils.test(model, testloader, device=DEVICE)
        return loss, {"accuracy": accuracy}

    return evaluate


if __name__ == "__main__":
    main()
