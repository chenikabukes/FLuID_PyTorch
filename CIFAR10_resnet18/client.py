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
from torchmetrics.clustering import MutualInfoScore
import torchvision
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, ParametersRes, Weights
import numpy as np
import matplotlib.pyplot as plt
from mine import MINE
from sklearn.feature_selection import mutual_info_regression

#import ssl
#ssl._create_default_https_context = ssl._create_unverified_context

import utils

# pylint: disable=no-member
#DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# pylint: enable=no-member
# Generates data by sampling from two correlated Gaussian variables
# dim = 1
# variance = 0.2
# sampleSize = 2000

# xSamples = np.sign(np.random.normal(0., 1., [sampleSize, dim]))
# ySamples = xSamples + np.random.normal(0., np.sqrt(variance), [sampleSize, dim])
# pyx = np.exp(-(ySamples - xSamples) ** 2 / (2 * variance))
# pyxMinus = np.exp(-(ySamples + 1) ** 2 / (2 * variance))
# pyxPlus = np.exp(-(ySamples - 1) ** 2 / (2 * variance))

# mi = np.average(np.log(pyx / (0.5 * pyxMinus + 0.5 * pyxPlus)))

# miEstimator = MINE(dim, archSpecs={
#     'layerSizes': [32] * 1,
#     'activationFunctions': ['relu'] * 1
# }, divergenceMeasure='KL', learningRate=1e-3)


# ySamplesMarginal = np.random.permutation(ySamples)

# estimatedMI, estimationHistory = miEstimator.calcMI(xSamples, ySamples, xSamples, ySamplesMarginal,
#                                                     batchSize=sampleSize, numEpochs=2000)

# print("Real MI: {}, estimated MI: {}".format(mi, estimatedMI))
# print("Estimated MI: {}".format(estimatedMI))
# epochs = np.arange(len(estimationHistory))
# plt.plot(epochs, estimationHistory)
# plt.plot(epochs, mi * np.ones(len(estimationHistory)))
# plt.xlabel('Epochs')
# plt.ylabel('Estimated MI')
# plt.legend(['Estimated', 'Real'])
# plt.show()

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
        utils.train(self.model, trainloader, epochs=epochs, device=self.device)
        fitTime = time() - t

        # Return the refined weights and the number of examples used for training
        weights_prime: Weights = get_weights(self.model)
        params_prime = fl.common.weights_to_parameters(weights_prime)
        num_examples_train = len(self.trainset)
        metrics = {"duration": timeit.default_timer() - fit_begin}
        return FitRes(
            parameters=params_prime, num_examples=num_examples_train, metrics=metrics, fit_duration=fitTime
        )

    # def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
    #     print(f"Client {self.cid}: evaluate")

    #     weights = fl.common.parameters_to_weights(ins.parameters)

    #     # Use provided weights to update the local model
    #     set_weights(self.model, weights)
    #     set_weights(self.model, weights)

    #     # Evaluate the updated model on the local dataset
    #     testloader = torch.utils.data.DataLoader(
    #         self.testset, batch_size=32, shuffle=False
    #     )
    #     loss, accuracy = utils.test(self.model, testloader, device=self.device)

    #     # Return the number of evaluation examples and the evaluation result (loss)
    #     metrics = {"accuracy": float(accuracy)}
    #     return EvaluateRes(
    #         num_examples=len(self.testset), loss=float(loss), metrics=metrics
    #     )
    # def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
    #     print(f"Client {self.cid}: evaluate")

    #     weights = fl.common.parameters_to_weights(ins.parameters)

    #     # Use provided weights to update the local model
    #     set_weights(self.model, weights)

    #     # Evaluate the updated model on the local dataset
    #     testloader = torch.utils.data.DataLoader(self.testset, batch_size=32, shuffle=False)
    #     loss, accuracy = utils.test(self.model, testloader, device=self.device)

    #     # Prepare a batch of data for MI calculation
    #     with torch.no_grad():  # Ensure no gradients are computed during evaluation
    #         batch_joint = next(iter(testloader))
    #         xSamplesJoint, labelsJoint = batch_joint
    #         xSamplesJoint = xSamplesJoint.to(self.device)
    #         outputsJoint = self.model(xSamplesJoint).cpu().detach().numpy()
    #         labelsJoint = labelsJoint.cpu().detach().numpy()

    #     # Optionally, fetch a different batch for the marginal distribution or shuffle xSamplesJoint
    #     xSamplesMarginal = np.random.permutation(outputsJoint)  # This is a simple approach to break the dependency

    #     # Labels as ySamples and outputs as xSamples in this context
    #     labelsJoint_reshaped = labelsJoint[:, np.newaxis]  
    #     ySamplesMarginal = np.random.permutation(labelsJoint_reshaped)
    #     print("shape of labelsJoint_reshaped", labelsJoint_reshaped.shape)
    #     print("shape of xSamplesJoint", xSamplesJoint.shape)
    #     print("shape of ySamplesMarginal", ySamplesMarginal.shape)
    #     print("shape of xSamplesMarginal", xSamplesMarginal.shape)
    #     # Calculate mutual information using MINE
    #     miEstimation = self.mine.calcMI(outputsJoint, labelsJoint_reshaped, xSamplesMarginal, ySamplesMarginal, batchSize=32, numEpochs=200)

    #     # Add MI estimation to metrics
    #     metrics = {"accuracy": float(accuracy), "mutual_info": miEstimation}

    #     return EvaluateRes(num_examples=len(self.testset), loss=float(loss), metrics=metrics)

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

    #     # Then, call estimate_mi with outputsJoint and labelsJoint as data
    #     mi_estimate, log = estimate_mi(
    #         data=(outputsJoint, labelsJoint),  # Pass the model outputs and labels as data
    #         estimator_name='js',  # The mutual information estimator to use
    #         hidden_dims=[32, 32],  # The hidden layers for the neural architectures
    #         neg_samples=16,  # The number of negative samples used
    #         batch_size=128,  # The batch size used for training
    #         max_epochs=500,  # Number of maximum training epochs
    #         valid_percentage=0.1,  # The percentage of data to use for validation
    #         evaluation_batch_size=256,  # The batch size used for evaluation
    #         device='cpu',  # The training device
    #     )


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


    # def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
    #     print(f"Client {self.cid}: evaluate")

    #     weights = fl.common.parameters_to_weights(ins.parameters)
    #     set_weights(self.model, weights)  # Use provided weights to update the local model

    #     testloader = torch.utils.data.DataLoader(self.testset, batch_size=32, shuffle=False)
    #     loss, accuracy = utils.test(self.model, testloader, device=self.device)  # Evaluate the updated model

    #     # Initialize the Mutual Information scorer
    #     mi_score = MutualInfoScore()

    #     # Prepare a batch of data for MI calculation
    #     with torch.no_grad():  # Ensure no gradients are computed during evaluation
    #         for batch in testloader:
    #             xSamplesJoint, labelsJoint = batch
    #             xSamplesJoint = xSamplesJoint.to(self.device)
    #             outputs = self.model(xSamplesJoint).argmax(dim=1).cpu()  

    #             # Calculate MI for the batch and accumulate
    #             mi_score.update(outputs, labelsJoint)

    #     # Finalize MI score calculation
    #     final_mi_score = mi_score.compute()

    #     print(f"Estimated Mutual Information: {final_mi_score.item()} nats")

    #     metrics = {"accuracy": float(accuracy), "mutual_info": final_mi_score.item()}  # Add MI estimation to metrics

    #     return EvaluateRes(num_examples=len(self.testset), loss=float(loss), metrics=metrics)


    # def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
    #     print(f"Client {self.cid}: evaluate")

    #     weights = fl.common.parameters_to_weights(ins.parameters)
    #     set_weights(self.model, weights)  # Use provided weights to update the local model

    #     testloader = torch.utils.data.DataLoader(self.testset, batch_size=32, shuffle=False)
    #     loss, accuracy = utils.test(self.model, testloader, device=self.device) 

    #     # Initialize the Mutual Information scorer
    #     mi_score = MutualInfoScore()
    #     mi_values = []

    #     # Prepare data for MI calculation
    #     with torch.no_grad():  # Ensure no gradients are computed during evaluation
    #         for batch in testloader:
    #             xSamplesJoint, labelsJoint = batch
    #             xSamplesJoint = xSamplesJoint.to(self.device)
    #             outputs = self.model(xSamplesJoint).argmax(dim=1).cpu()
    #             # Update the MI scorer with predictions and targets for this batch
    #             mi_score.update(outputs, labelsJoint)
    #             batch_mi = mi_score(outputs, labelsJoint).item()
    #             mi_values.append(batch_mi)

    #     # Compute the overall MI after all batches have been processed
    #     final_mi_score = mi_score.compute()

    #     print(f"Estimated Mutual Information: {final_mi_score.item()} nats")

    #     # Plotting the MI values
    #     epochs = list(range(1, len(mi_values) + 1))
    #     plt.plot(epochs, mi_values, marker='o', linestyle='-')
    #     plt.title(f'Mutual Information Over Epochs - Client {self.cid}')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Mutual Information')
    #     plt.grid(True)

    #     current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #     filename = f'./MIPlots/mutual_information_client_{self.cid}_{current_time}.png'

    #     plt.savefig(filename)  
    #     plt.close() 

    #     metrics = {"accuracy": float(accuracy), "mutual_info": final_mi_score.item()}  

    #     return EvaluateRes(num_examples=len(self.testset), loss=float(loss), metrics=metrics)


    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        print(f"Client {self.cid}: evaluate")

        weights = fl.common.parameters_to_weights(ins.parameters)
        set_weights(self.model, weights)  # Use provided weights to update the local model

        testloader = torch.utils.data.DataLoader(self.testset, batch_size=32, shuffle=False)
        loss, accuracy = utils.test(self.model, testloader, device=self.device) 

        # Initialize the Mutual Information scorer
        mi_score = MutualInfoScore()
        mi_values = []

        # Prepare data for MI calculation
        with torch.no_grad():  # Ensure no gradients are computed during evaluation
            for batch in testloader:
                xSamplesJoint, labelsJoint = batch
                xSamplesJoint = xSamplesJoint.to(self.device)
                outputs = self.model(xSamplesJoint).argmax(dim=1).cpu()
                # Update the MI scorer with predictions and targets for this batch
                mi_values.append(mi_score(outputs, labelsJoint))

        # Compute the overall MI after all batches have been processed
        fig_, ax_ = mi_score.plot(mi_values)
        fig_.savefig(f'./MIPlots/mutual_information_plot_client_{self.cid}.png')
        plt.show()
        final_mi_score = mi_score.compute()

        print(f"Estimated Mutual Information: {final_mi_score.item()} nats")

        metrics = {"accuracy": float(accuracy), "mutual_info": final_mi_score.item()}  

        return EvaluateRes(num_examples=len(self.testset), loss=float(loss), metrics=metrics)




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
