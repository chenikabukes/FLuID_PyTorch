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
"""Flower client example using PyTorch for CIFAR-10 image classification."""


import argparse
import pandas as pd
from torch_mist import estimate_mi
import matplotlib.pyplot as plt
import seaborn as sns
import timeit
from collections import OrderedDict
import datetime
from importlib import import_module

import flwr as fl
import numpy as np
from time import time
import torch

import torchvision
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, ParametersRes, Weights
import numpy as np
import matplotlib.pyplot as plt
from mine import MINE
from sklearn.feature_selection import mutual_info_regression

import utils


def get_weights(model: torch.nn.ModuleList) -> fl.common.Weights:
    """Get model weights as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(model: torch.nn.ModuleList, weights: fl.common.Weights) -> None:
    """Set model weights from a list of NumPy ndarrays."""
    state_dict = OrderedDict(
        {
            k: torch.Tensor(np.atleast_1d(v))
            for k, v in zip(model.state_dict().keys(), weights)
        }
    )
    model.load_state_dict(state_dict, strict=True)

def match_dimensions_and_calculate_mi(gradients, inputs, sample_size=1024):
    """
    Randomly sample elements from gradients and inputs to match dimensions and calculate MI.
    sample_size: Number of elements to sample, ensuring it's less than min(input elements, gradient elements)
    """
    inputs_flat = inputs.view(-1)  # [16*3*32*32]
    
    # Ensure sample_size is feasible
    sample_size = min(sample_size, inputs_flat.size(0), gradients.size(0))
    
    # Randomly sample indices
    indices = np.random.choice(inputs_flat.size(0), sample_size, replace=False)
    
    # Sample from inputs and gradients
    inputs_sample = inputs_flat[torch.tensor(indices)].cpu().numpy()
    gradients_sample = gradients[torch.tensor(indices)].cpu().numpy()
    
    mi_estimate, log = estimate_mi(
            data=(inputs_sample, gradients_sample),  
            estimator_name='js',  
            hidden_dims=[32, 32], 
            neg_samples=16,
            batch_size=128,
            max_epochs=500,
            valid_percentage=0.1,
            evaluation_batch_size=256,
            device='cpu',  
        )
    
    return mi_estimate  

class CifarClient(fl.client.Client):
    """Flower client implementing CIFAR-10 image classification using
    PyTorch."""

    def __init__(
        self,
        cid: str,
        model: torch.nn.Module,
        trainset: torchvision.datasets.CIFAR10,
        testset: torchvision.datasets.CIFAR10,
        device: torch.device,
    ) -> None:
        self.cid = cid
        self.model = model
        self.trainset = trainset
        self.testset = testset
        self.device = device
        self.p = 1
        model_output_dim = model.fc.out_features  
        self.mine = MINE(inputSpaceDim=model_output_dim, archSpecs={
            'layerSizes': [32] * 1,
            'activationFunctions': ['relu'] * 1
        }, divergenceMeasure='KL', learningRate=1e-3)
        self.mi_values = []

    def get_parameters(self) -> ParametersRes:
        print(f"Client {self.cid}: get_parameters")

        weights: Weights = get_weights(self.model)
        parameters = fl.common.weights_to_parameters(weights)
        return ParametersRes(parameters=parameters)

    def _instantiate_model(self, model_str: str):

        # will load utils.model_str
        m = getattr(import_module("utils"), model_str)
        # instantiate model
        self.model = m()

    # def fit(self, ins: FitIns) -> FitRes:
    #     print(f"Client {self.cid}: fit")

    #     weights: Weights = fl.common.parameters_to_weights(ins.parameters)
    #     config = ins.config
    #     fit_begin = timeit.default_timer()

    #     # Get training config
    #     epochs = int(config["epochs"])
    #     batch_size = int(config["batch_size"])
    #     pin_memory = bool(config["pin_memory"])
    #     num_workers = int(config["num_workers"])

    #     # fix_for_drop
    #     p = float(config["p"])
    #     if (p != self.p):
    #         print("changing p from " + str(self.p) + " to " + str(p))
    #         self.p = p
    #         self.model = utils.load_model("ResNet18", self.p)
    #         self.model.to(self.device)

    #     # Set model parameters
    #     set_weights(self.model, weights)

    #     if torch.cuda.is_available():
    #         kwargs = {
    #             "num_workers": num_workers,
    #             "pin_memory": pin_memory,
    #             "drop_last": True,
    #         }
    #     else:
    #         kwargs = {"drop_last": True}

        # # Train model
        # trainloader = torch.utils.data.DataLoader(
        #     self.trainset, batch_size=batch_size, shuffle=True, **kwargs
        # )
        # t = time()
        # utils.train(self.model, trainloader, epochs=epochs, device=self.device)
        # fitTime = time() - t

    #     # Return the refined weights and the number of examples used for training
    #     weights_prime: Weights = get_weights(self.model)
    #     params_prime = fl.common.weights_to_parameters(weights_prime)
    #     num_examples_train = len(self.trainset)
    #     metrics = {"duration": timeit.default_timer() - fit_begin}
    #     return FitRes(
    #         parameters=params_prime, num_examples=num_examples_train, metrics=metrics, fit_duration=fitTime
    #     )

    def fit(self, ins: FitIns) -> FitRes:
        print(f"Client {self.cid}: fit")

        weights: Weights = fl.common.parameters_to_weights(ins.parameters)
        config = ins.config
        fit_begin = timeit.default_timer()

        # Get training config
        epochs = int(config["epochs"])
        batch_size = int(config["batch_size"])
        pin_memory = bool(config["pin_memory"])
        num_workers = int(config["num_workers"])

        # fix_for_drop
        p = float(config["p"])
        if (p != self.p):
            print("changing p from " + str(self.p) + " to " + str(p))
            self.p = p
            self.model = utils.load_model("ResNet18", self.p)
            self.model.to(self.device)

        # Set model parameters
        set_weights(self.model, weights)

        if torch.cuda.is_available():
            kwargs = {
                "num_workers": num_workers,
                "pin_memory": pin_memory,
                "drop_last": True,
            }
        else:
            kwargs = {"drop_last": True}

            # Train model
        trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=batch_size, shuffle=True, **kwargs
        )
        t = time()
        gradients, representative_inputs = utils.train_mi(self.model, trainloader, epochs=epochs, device=self.device)
        print(gradients.shape)
        print(representative_inputs.shape)
        fitTime = time() - t

        # Calculate mutual information
        mi_estimate = match_dimensions_and_calculate_mi(gradients, representative_inputs)
        print(f"Client {self.cid}: Mutual Information after fit round: {mi_estimate}")
        self.mi_values.append(mi_estimate)
        # Return the refined weights and the number of examples used for training
        weights_prime: Weights = get_weights(self.model)
        params_prime = fl.common.weights_to_parameters(weights_prime)
        num_examples_train = len(self.trainset)
        metrics = {"duration": timeit.default_timer() - fit_begin}
        # Include MI in the metrics
        metrics.update({"mutual_information": mi_estimate})
        return FitRes(
            parameters=params_prime, num_examples=num_examples_train, metrics=metrics, fit_duration=fitTime
        )
        
    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        print(f"Client {self.cid}: evaluate")

        weights = fl.common.parameters_to_weights(ins.parameters)

        # Use provided weights to update the local model
        set_weights(self.model, weights)
        set_weights(self.model, weights)

        # Evaluate the updated model on the local dataset
        testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=32, shuffle=False
        )
        loss, accuracy = utils.test(self.model, testloader, device=self.device)

        # Return the number of evaluation examples and the evaluation result (loss)
        metrics = {"accuracy": float(accuracy)}
        return EvaluateRes(
            num_examples=len(self.testset), loss=float(loss), metrics=metrics
        )

    # def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
    #     print(f"Client {self.cid}: evaluate")

    #     weights = fl.common.parameters_to_weights(ins.parameters)
    #     set_weights(self.model, weights)  # Use provided weights to update the local model

    #     testloader = torch.utils.data.DataLoader(self.testset, batch_size=32, shuffle=False)
    #     loss, accuracy = utils.test(self.model, testloader, device=self.device)  # Evaluate the updated model

    #     with torch.no_grad():  # Prepare data for MI calculation
    #         batch_joint = next(iter(testloader))
    #         xSamplesJoint, labelsJoint = batch_joint
    #         xSamplesJoint = xSamplesJoint.to(self.device)
    #         outputsJoint = self.model(xSamplesJoint).cpu().detach().numpy()  # Model outputs as x
    #         labelsJoint = labelsJoint.cpu().detach().numpy()  # Corresponding labels as y

        # # Then, call estimate_mi with outputsJoint and labelsJoint as data
        # mi_estimate, log = estimate_mi(
        #     data=(outputsJoint, labelsJoint),  # Pass the model outputs and labels as data
        #     estimator_name='js',  # The mutual information estimator to use
        #     hidden_dims=[32, 32],  # The hidden layers for the neural architectures
        #     neg_samples=16,  # The number of negative samples used
        #     batch_size=128,  # The batch size used for training
        #     max_epochs=500,  # Number of maximum training epochs
        #     valid_percentage=0.1,  # The percentage of data to use for validation
        #     evaluation_batch_size=256,  # The batch size used for evaluation
        #     device='cpu',  # The training device
        # )


    #     print(f"Estimated Mutual Information: {mi_estimate} nats")

    #     metrics = {"accuracy": float(accuracy), "mutual_info": mi_estimate}  

    #     # Convert log to a DataFrame if it's not already one
    #     if not isinstance(log, pd.DataFrame):
    #         log_df = pd.DataFrame(log)
    #     else:
    #         log_df = log

    #     # Plotting the estimated values by epoch
    #     grid = sns.FacetGrid(log_df, col='name', hue='split', sharey=False, col_order=['loss', 'mutual_information'])
    #     grid.map(sns.lineplot, 'epoch', 'value')
    #     grid.add_legend()
    #     grid.set_titles(col_template='{col_name}')

    #      # Save the plot to a file
    #     plt.savefig('./MIPlots/mutual_information_plot.png')  

    #     plt.close('all')  

    #     return EvaluateRes(num_examples=len(self.testset), loss=float(loss), metrics=metrics)


    def plot_mutual_information(client):
        plt.figure(figsize=(8, 6))
        plt.plot(range(len(client.mi_values)), client.mi_values, marker='o')
        plt.title(f'Mutual Information - Client {client.cid}')
        plt.xlabel('Fit Round')
        plt.ylabel('Mutual Information')
        plt.show()

def main() -> None:
    """Load data, create and start CifarClient."""
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--server_address",
        type=str,
        required=True,
        help=f"gRPC server address",
    )
    parser.add_argument(
        "--cid", type=str, required=True, help="Client CID (no default)"
    )
    parser.add_argument(
        "--log_host",
        type=str,
        help="Logserver address (no default)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory where the dataset lives",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Net",
        choices=["Net", "ResNet18"],
        help="model to train",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="gpu",
        choices=["gpu", "cpu"],
        help="processor to run client on",
    )
    parser.add_argument(
        "--device_idx",
        type=int,
        default=0,
        help="processor to run client on",
    )


    args = parser.parse_args()

    # Configure logger
    fl.common.logger.configure(f"client_{args.cid}", host=args.log_host)

    # model
    model = utils.load_model(args.model)
    if (args.device == "cpu"):
        device = torch.device("cpu", args.device_idx )
        print ("running on CPU")
    elif (torch.cuda.is_available()):
        print ("running on GPU")
        device = torch.device("cuda",args.device_idx)
    else:
        print ("GPU unavailble, running on CPU")
        device = torch.device("cpu", args.device_idx)
    model.to(device)
    # load (local, on-device) dataset
    trainset, testset = utils.load_dataset(args.model, args.cid)

    # Start client
    client = CifarClient(args.cid, model, trainset, testset, device)

    fl.client.start_client(args.server_address, client)


if __name__ == "__main__":
    main()
