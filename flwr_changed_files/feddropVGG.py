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
"""Federated Averaging (FedAvg) [McMahan et al., 2016] strategy.

Paper: https://arxiv.org/abs/1602.05629
"""


from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    Weights,
    parameters_to_weights,
    weights_to_parameters,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from .aggregate import aggregate, aggregate_drop, weighted_loss_avg
from .strategy import Strategy
import numpy as np
import random

DEPRECATION_WARNING = """
DEPRECATION WARNING: deprecated `eval_fn` return format

    loss, accuracy

move to

    loss, {"accuracy": accuracy}

instead. Note that compatibility with the deprecated return format will be
removed in a future release.
"""

DEPRECATION_WARNING_INITIAL_PARAMETERS = """
DEPRECATION WARNING: deprecated initial parameter type

    flwr.common.Weights (i.e., List[np.ndarray])

will be removed in a future update, move to

    flwr.common.Parameters

instead. Use

    parameters = flwr.common.weights_to_parameters(weights)

to easily transform `Weights` to `Parameters`.
"""

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_eval_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_eval_clients`.
"""


class FedDropVGG(Strategy):
    """Configurable FedAvg strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        fraction_fit: float = 0.1,
        fraction_eval: float = 0.1,
        min_fit_clients: int = 1,
        min_eval_clients: int = 1,
        min_available_clients: int = 1,
        eval_fn: Optional[
            Callable[[Weights], Optional[Tuple[float, Dict[str, Scalar]]]]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
    ) -> None:
        """Federated Averaging strategy.

        Implementation based on https://arxiv.org/abs/1602.05629

        Parameters
        ----------
        fraction_fit : float, optional
            Fraction of clients used during training. Defaults to 0.1.
        fraction_eval : float, optional
            Fraction of clients used during validation. Defaults to 0.1.
        min_fit_clients : int, optional
            Minimum number of clients used during training. Defaults to 2.
        min_eval_clients : int, optional
            Minimum number of clients used during validation. Defaults to 2.
        min_available_clients : int, optional
            Minimum number of total clients in the system. Defaults to 2.
        eval_fn : Callable[[Weights], Optional[Tuple[float, Dict[str, Scalar]]]]
            Optional function used for validation. Defaults to None.
        on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure training. Defaults to None.
        on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure validation. Defaults to None.
        accept_failures : bool, optional
            Whether or not accept rounds containing failures. Defaults to True.
        initial_parameters : Parameters, optional
            Initial global model parameters.
        """
        super().__init__()

        if (
            min_fit_clients > min_available_clients
            or min_eval_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.fraction_fit = fraction_fit
        self.fraction_eval = fraction_eval
        self.min_fit_clients = min_fit_clients
        self.min_eval_clients = min_eval_clients
        self.min_available_clients = min_available_clients
        self.eval_fn = eval_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.droppedWeights: Dict[str, List] = {}
        self.straggler: Dict[str, float] = {}
        self.p_val: Dict[str, float] = {}
        self.unchagedWeights = [[] for x in range(18)]
        self.defDropWeights = [[] for x in range(18)]
        self.prevDropWeights = [[] for x in range(18)]
        self.changeThreshold = [1.5 for x in range (18)]
        self.changeIncrement = 0.2
        self.roundCounter = 0
        self.stopChange = [False for x in range (18)]
        self.parameters: Parameters
        self.constant_pval = 0.75

    def __repr__(self) -> str:
        rep = f"FedAvg(accept_failures={self.accept_failures})"
        return rep

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return the sample size and the required number of available
        clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_eval)
        return max(num_clients, self.min_eval_clients), self.min_available_clients

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""

        random_client = client_manager.sample(1)[0]
        self.initial_parameters = random_client.get_parameters().parameters
        initial_parameters = self.initial_parameters
        #self.initial_parameters = None  # Don't keep initial parameters in memory
        if isinstance(initial_parameters, list):
            log(WARNING, DEPRECATION_WARNING_INITIAL_PARAMETERS)
            initial_parameters = weights_to_parameters(weights=initial_parameters)
        return initial_parameters

    def evaluate(
        self, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.eval_fn is None:
            # No evaluation function provided
            return None
        weights = parameters_to_weights(parameters)
        eval_res = self.eval_fn(weights)
        if eval_res is None:
            return None
        loss, other = eval_res
        if isinstance(other, float):
            print(DEPRECATION_WARNING)
            metrics = {"accuracy": other}
        else:
            metrics = other
        return loss, metrics

    def configure_fit(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        self.parameters = parameters
        config = {}
        config_drop = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(rnd)

        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        clientList = []
        for client in clients:
            if (client.cid in self.straggler) and rnd > 1:
                p_val = self.p_val[client.cid]
                #print(p_val)
                config_drop = self.on_fit_config_fn(rnd,p_val)

                # CHANGE HERE for each dropout method. See method for clarification on each input to the method
                # drop_rand - random dropout
                # drop_order - ordered dropout
                # drop_dynamic - invariant dropout

                fit_ins_drop = FitIns(self.drop_order(parameters, p_val, [0,2,4,6,8,12,14], 10, client.cid), config_drop)
                clientList.append((client, fit_ins_drop))
            else:
                clientList.append((client, fit_ins))
        return clientList

    def configure_evaluate(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if a centralized evaluation
        # function is provided
        if self.eval_fn is not None:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(rnd)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        if rnd >= 0:
            sample_size, min_num_clients = self.num_evaluation_clients(
                client_manager.num_available()
            )
            clients = client_manager.sample(
                num_clients=sample_size, min_num_clients=min_num_clients
            )
        else:
            clients = list(client_manager.all().values())

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        # Convert results
        weights_results = [
            (parameters_to_weights(fit_res.parameters), fit_res.num_examples, client.cid)
            for client, fit_res in results
        ]

        # Set different threshold values for each layer
        idxList = [0,2,4,6,8,12,14]
        if (rnd > 10):
            self.roundCounter += 1
            if ( self.roundCounter >= 2): 
                for idx in idxList:
                    if self.stopChange[idx] != True:
                        self.changeThreshold[idx] += self.changeIncrement
                        self.roundCounter = 0
                        print("threshold for ", idx, " updated to: ", self.changeThreshold[idx])

        # find invariant neuron
        self.find_stable(self.parameters, weights_results,[0,2,4,6,8,12,14], 10)

        # initialize thresholds (only done once) 
        self.find_min(self.parameters, weights_results,[0,2,4,6,8,12,14], rnd)
        aggregated_weights = aggregate_drop(weights_results, self.droppedWeights, parameters_to_weights(self.initial_parameters))


        def time(elem):
                return elem[1].fit_duration

        results.sort(key=time)

        # set number of stragglers 
        numStrag = int(len(results) * 0.2)
        #numStrag = 40
        # numInClass = int(numStrag * 0.25)

         # first time adding stragglers to the list
        if (len(self.straggler) == 0 and rnd > 1):
            target = results[len(results) - 1 - numStrag][1].fit_duration
            for i in range(numStrag):
                newStrag = results[len(results) - 1 - i]
                self.straggler[newStrag[0].cid] = newStrag[1].fit_duration


                # set sub-model size of all stragglers to be the same. 
                # If we want to dynamically decide the p_val, use the "target" variable to get the next slowest client
                self.p_val[newStrag[0].cid] = self.constant_pval

                # divide stragglers in to equal partitions for different sub-model sizes

                #if (i < numInClass):
                #    self.p_val[newStrag[0].cid] = 0.65
                #elif (i < (2* numInClass + 1) ):
                #    self.p_val[newStrag[0].cid] = 0.75
                #elif (i < (3* numInClass + 2)):
                #    self.p_val[newStrag[0].cid] = 0.85
                #else:
                #    self.p_val[newStrag[0].cid] = 0.95
                 
            self.straggler = dict(sorted(self.straggler.items(), key=lambda item: item[1]))
            print(self.straggler)
            print(self.p_val)

        # continue to update list and check if there are any changes in stragglers
        elif (rnd > 1):
            stragglerList = list(self.straggler.items())
            for i in range(numStrag):
                slowest = results[len(results) - 1 - i]
                if slowest[0].cid in self.straggler:
                    continue
                # only swap if new slowest client is 10% slower than "fastest" straggler
                elif slowest[1].fit_duration > (stragglerList[0][1]* 1.1):
                    self.straggler[slowest[0].cid] = slowest[1].fit_duration
                    self.straggler.pop(stragglerList[0][0])
                    self.droppedWeights.pop(stragglerList[0][0])
                    self.p_val.pop(stragglerList[0][0])
                    stragglerList.pop(0)
                    print("swapped straggler")
                else:
                    break
            self.straggler = dict(sorted(self.straggler.items(), key=lambda item: item[1]))
            stragglerList = list(self.straggler.items())
            for i in range(numStrag):

                # set the p value for the new straggler
                self.p_val[stragglerList[i][0]] = self.constant_pval

                #if (i < numInClass):
                #    self.p_val[newStrag[0].cid] = 0.65
                #elif (i < (2* numInClass + 1) ):
                #    self.p_val[newStrag[0].cid] = 0.75
                #elif (i < (3* numInClass + 2)):
                #    self.p_val[newStrag[0].cid] = 0.85
                #else:
                #    self.p_val[newStrag[0].cid] = 0.95

            print(self.straggler)
            print(self.p_val)

        return weights_to_parameters(aggregated_weights), {}

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        loss_aggregated = weighted_loss_avg(
            [
                (
                    evaluate_res.num_examples,
                    evaluate_res.loss,
                    evaluate_res.accuracy,
                )
                for _, evaluate_res in results
            ]
        )
        return loss_aggregated, {}


    # random dropout

    # p - sub-model size 

    # idxList - list of indicies in the weights array indicating the first set/layer of parameters in each dropout transformation
    # note, idx + 1 is usually the bias parameters of the layer

    # Each dropout transformation affects 3 idx of parameters in the weights array 
    # (idx = weights @ dimension 0, idx + 1 = bias @ dimension 0, idx + 2 = weights (of the next layer) @ dimension 1 )
    # we need to transform all "layers" of parameters

    # idxConvFC - represents the convolutional layer that is followed by a fully connected layer 
    # we need to transform the layer while taking into account that the layer shape has changed

    # cid - id of the client

    def drop_rand(self, parameters: Parameters, p: float, idxList: List[int], idxConvFC: int, cid:str):
        weights = parameters_to_weights(parameters)
        if cid not in self.droppedWeights:
            self.droppedWeights[cid] = [[[],[]] for x in range(len(weights))]

        for idx in idxList:
            shape = weights[idx].shape
            list = random.sample(range(1, shape[0]), shape[0] - int(p * shape[0]))
            list.sort()
            self.droppedWeights[cid][idx][0] = list.copy()
            self.droppedWeights[cid][idx+1][0] = list.copy()
            self.droppedWeights[cid][idx+2][1] = list.copy()

            self.prevDropWeights[idx] = list.copy()
            print("Dropped weights idx ", idx, ": ", (self.prevDropWeights[idx]))

            # remove each row/column from the back
            weights[idx] = np.delete(weights[idx], list, 0)
            weights[idx + 1] = np.delete(weights[idx + 1], list, 0)
            weights[idx + 2] = np.delete(weights[idx + 2], list, 1)


        shape = weights[idxConvFC].shape
        listFC = random.sample(range(1, shape[0]), shape[0] - int(p * shape[0]))
        listFC.sort()
        FClist = []
        for drop in listFC:
            FClist.extend(range(drop*4*4, drop*4*4 + 4*4))

        self.droppedWeights[cid][idxConvFC][0] = listFC.copy()
        self.droppedWeights[cid][idxConvFC + 1][0] = listFC.copy()
        self.droppedWeights[cid][idxConvFC + 2][1] = FClist.copy()

        self.prevDropWeights[idxConvFC] = listFC.copy()
        print("Dropped weights idx ", idxConvFC, ": ", (self.prevDropWeights[idxConvFC]))

        weights[idxConvFC] = np.delete(weights[idxConvFC], listFC, 0)
        weights[idxConvFC + 1] = np.delete(weights[idxConvFC + 1], listFC, 0)
        weights[idxConvFC + 2] = np.delete(weights[idxConvFC + 2], FClist, 1)

        return weights_to_parameters(weights)

    # ordered dropout

    # p - sub-model size 

    # idxList - list of indicies in the weights array indicating the first set/layer of parameters in each dropout transformation
    # note, idx + 1 is usually the bias parameters of the layer

    # Each dropout transformation affects 3 idx of parameters in the weights array 
    # (idx = weights @ dimension 0, idx + 1 = bias @ dimension 0, idx + 2 = weights (of the next layer) @ dimension 1 )
    # we need to transform all "layers" of parameters

    # idxConvFC - represents the convolutional layer that is followed by a fully connected layer 
    # we need to transform the layer while taking into account that the layer shape has changed

    # cid - id of the client

    def drop_order(self, parameters: Parameters, p: float, idxList: List[int], idxConvFC: int, cid:str):
        weights = parameters_to_weights(parameters)
        if cid not in self.droppedWeights:
            self.droppedWeights[cid] = [[[],[]] for x in range(len(weights))]

        for idx in idxList:
            shape = weights[idx].shape
            numToDrop = shape[0] - int(p * shape[0])
            list = [x for x in range(shape[0] - numToDrop, shape[0])]

            self.droppedWeights[cid][idx][0] = list.copy()
            self.droppedWeights[cid][idx+1][0] = list.copy()
            self.droppedWeights[cid][idx+2][1] = list.copy()

            self.prevDropWeights[idx] = list.copy()
            print("Dropped weights idx ", idx, ": ", (self.prevDropWeights[idx]))

            # remove each row/column from the back
            weights[idx] = np.delete(weights[idx], list, 0)
            weights[idx + 1] = np.delete(weights[idx + 1], list, 0)
            weights[idx + 2] = np.delete(weights[idx + 2], list, 1)


        shape = weights[idxConvFC].shape
        numToDrop = shape[0] - int(p * shape[0])
        listFC = [x for x in range(shape[0] - numToDrop, shape[0])]
        FClist = []
        for drop in listFC:
            FClist.extend(range(drop*4*4, drop*4*4 + 4*4))

        self.droppedWeights[cid][idxConvFC][0] = listFC.copy()
        self.droppedWeights[cid][idxConvFC + 1][0] = listFC.copy()
        self.droppedWeights[cid][idxConvFC + 2][1] = FClist.copy()

        self.prevDropWeights[idxConvFC] = listFC.copy()
        print("Dropped weights idx ", idxConvFC, ": ", (self.prevDropWeights[idxConvFC]))

        weights[idxConvFC] = np.delete(weights[idxConvFC], listFC, 0)
        weights[idxConvFC + 1] = np.delete(weights[idxConvFC + 1], listFC, 0)
        weights[idxConvFC + 2] = np.delete(weights[idxConvFC + 2], FClist, 1)

        return weights_to_parameters(weights)

    # Find invariant neurons
    def find_stable(self, parameters: Parameters, results: List[Tuple[Weights, int, str]], idxList: List[int], idxConvFC: int):

        weights = parameters_to_weights(parameters)

        list = [[] for x in range(len(weights))]
        for idx in idxList: 

            difference = []
            for i in range(len(weights)):

                difference.append(np.full(weights[i].shape, 0))

            for cWeights, num_examples, cid in results:
                if cid in self.straggler:
                    continue
                clientWeights = cWeights

                for i in range(len(weights)):
                    difference[i] += (np.abs(clientWeights[i] - weights[i]) <= np.abs(self.changeThreshold[idx]* weights[i])) * 1
               
            for i in range(len(difference)):
                difference[i] = difference[i] > (0.75 * (len(results) - len(self.straggler)))


            #list[0] = self.unchagedWeights[0]

            shape = weights[idx].shape
            dim = [x for x in range(weights[idx].ndim)]
            dim.remove(0)

            # perform reduction for all other dimensions (so we know which idx has constant weights
            idx0Layer =  np.all(difference[idx], axis=tuple(dim))

            idx1Layer = difference[idx +1]

            dim = [x for x in range(weights[idx+2].ndim)]
            dim.remove(1)
            idx2Layer = np.all(difference[idx+2], axis=tuple(dim))

            noChangeIdx =  idx0Layer & idx1Layer & idx2Layer
            for i in range (len(noChangeIdx)):
                if noChangeIdx[i]:
                    if i not in list[idx]:
                        list[idx].append(i)
            print("unchanged idx ", idx, ": ", list[idx])


            self.defDropWeights[idx] = []
            if len(self.prevDropWeights[idx]) > 0:
                for i in self.prevDropWeights[idx]:
                    if i in list[idx]:
                        self.defDropWeights[idx].append(i)
            print("def drop idx ", idx, ": ", self.defDropWeights[idx])



        dim = [x for x in range(weights[idxConvFC].ndim)]
        dim.remove(0)
        idx0Layer = np.all(difference[idxConvFC], axis=tuple(dim))

        idx1Layer = difference[idxConvFC + 1]

        dim = [x for x in range(weights[idxConvFC + 2].ndim)]
        dim.remove(1)
        idx2Layer = np.all(difference[idxConvFC + 2], axis=tuple(dim))

        reducedList = np.array([])
        for i in range(len(idx0Layer)):
            reducedList = np.array(np.append(reducedList, np.all(idx2Layer[i*4*4 :  i * 4 * 4 + 4 * 4])), dtype=bool)
        print(len(reducedList))

        noChangeIdx = idx0Layer & idx1Layer & reducedList
        for i in range(len(noChangeIdx)):
            if noChangeIdx[i]:
                list[idxConvFC].append(i)
        print("unchanged idx ", idxConvFC, ": ", list[idxConvFC])

        self.defDropWeights[idxConvFC] = []
        if len(self.prevDropWeights[idxConvFC]) > 0:
            for i in self.prevDropWeights[idxConvFC]:
                if i in list[idxConvFC]:
                    self.defDropWeights[idxConvFC].append(i)
        print("def drop idx ", idxConvFC, ": ", self.defDropWeights[idxConvFC])

        self.unchagedWeights = list


        return list

     # find the minimum weight changes and initialize threshold
    def find_min(self, parameters: Parameters, results: List[Tuple[Weights, int, str]], idxList: List[int], rnd: int):
        weights = parameters_to_weights(parameters)
        list = []
        difference = []
        for i in range(len(weights)):
            difference.append(np.full(weights[i].shape, 0.0))

        for cWeights, num_examples, cid in results:
            if cid in self.straggler:
                continue
            clientWeights = cWeights

            for i in range(len(weights)):
                
                #get maximum % difference
                difference[i] = np.maximum(difference[i], (np.abs(clientWeights[i] - weights[i])) / np.abs(weights[i]))


        list = [[] for x in range(len(weights))]
        

        for idx in idxList:
            dim = [x for x in range(weights[idx].ndim)]
            dim.remove(0)

            # perform reduction for all other dimensions (so we know which idx has constant weights
            idx0Layer =  np.amax(difference[idx], axis=tuple(dim))

            idx1Layer = difference[idx +1]

            dim = [x for x in range(weights[idx+2].ndim)]
            dim.remove(1)
            idx2Layer = np.amax(difference[idx+2], axis=tuple(dim))
            sum = np.maximum(np.maximum(idx0Layer, idx1Layer),idx2Layer)

            noChangeIdx =  np.argsort(sum)


            print("% difference: ", sum[noChangeIdx[0]])
            if (rnd == 2):
                self.changeThreshold[idx] = sum[noChangeIdx[0]]
                print("threshold updated to: ", self.changeThreshold[idx])
            if (rnd == 3):
                self.changeThreshold[idx] = sum[noChangeIdx[0]]
                print("threshold updated to: ", self.changeThreshold[idx])

    # invariant dropout
                
    # p - sub-model size 

    # idxList - list of indicies in the weights array indicating the first set/layer of parameters in each dropout transformation
    # note, idx + 1 is usually the bias parameters of the layer

    # Each dropout transformation affects 3 idx of parameters in the weights array 
    # (idx = weights @ dimension 0, idx + 1 = bias @ dimension 0, idx + 2 = weights (of the next layer) @ dimension 1 )
    # we need to transform all "layers" of parameters

    # idxConvFC - represents the convolutional layer that is followed by a fully connected layer 
    # we need to transform the layer while taking into account that the layer shape has changed

    # cid - id of the client
                
    def drop_dynamic(self, parameters: Parameters, p: float, idxList: List[int], idxConvFC: int, cid: str):
        weights = parameters_to_weights(parameters)

        if cid not in self.droppedWeights:
            self.droppedWeights[cid] = [[[], []] for x in range(len(weights))]

        for idx in idxList:

            first = 0
            second = 0
            third = 1

            shape = weights[idx].shape
            numToDrop = shape[first] - int(p * shape[first])
            if len(self.unchagedWeights[idx]) >= numToDrop:
                self.stopChange[idx] = True

                fullList = self.unchagedWeights[idx].copy()
                for x in self.defDropWeights[idx]:
                    fullList.remove(x)
                if (len(self.defDropWeights[idx]) > numToDrop):
                    list = random.sample(self.defDropWeights[idx], numToDrop)
                else:
                    list = random.sample(fullList, numToDrop - len(self.defDropWeights[idx]))
                    list.extend(self.defDropWeights[idx])
                #list = random.sample(self.unchagedWeights[idx], numToDrop)
                list.sort()
            else:
                fullList = [x for x in range(shape[first])]
                for x in self.unchagedWeights[idx]:
                    fullList.remove(x)
                list = random.sample(fullList, numToDrop - len(self.unchagedWeights[idx]))
                list.extend(self.unchagedWeights[idx])
                list.sort()

            self.prevDropWeights[idx] = list.copy()
            print("Dropped weights idx ", idx, ": ", (self.prevDropWeights[idx]))

            self.droppedWeights[cid][idx][0] = list.copy()
            self.droppedWeights[cid][idx + 1][0] = list.copy()
            self.droppedWeights[cid][idx + 2][1] = list.copy()

            # remove each row/column from the back
            weights[idx] = np.delete(weights[idx], list, first)
            weights[idx + 1] = np.delete(weights[idx + 1], list, second)
            weights[idx + 2] = np.delete(weights[idx + 2], list, third)

        first = 0
        second = 0
        third = 1

        shape = weights[idxConvFC].shape
        
        numToDrop = shape[first] - int(p * shape[first])
        if len(self.unchagedWeights[idxConvFC]) >= numToDrop:
            listFC = random.sample(self.unchagedWeights[idxConvFC], numToDrop)
            listFC.sort()
        else:
            fullList = [x for x in range(shape[first])]
            for x in self.unchagedWeights[idxConvFC]:
                fullList.remove(x)
            listFC = random.sample(fullList, numToDrop - len(self.unchagedWeights[idxConvFC]))
            listFC.extend(self.unchagedWeights[idxConvFC])
            listFC.sort()
        self.prevDropWeights[idxConvFC] = listFC.copy()
        print("Dropped weights idx ", idxConvFC, ": ", (self.prevDropWeights[idxConvFC]))

        FClist = []
        for drop in listFC:
            FClist.extend(range(drop * 4 * 4, drop * 4 * 4 + 4 * 4))

        self.droppedWeights[cid][idxConvFC][0] = listFC.copy()
        self.droppedWeights[cid][idxConvFC + 1][0] = listFC.copy()
        self.droppedWeights[cid][idxConvFC + 2][1] = FClist.copy()

        weights[idxConvFC] = np.delete(weights[idxConvFC], listFC, first)
        weights[idxConvFC + 1] = np.delete(weights[idxConvFC + 1], listFC, second)
        weights[idxConvFC + 2] = np.delete(weights[idxConvFC + 2], FClist, third)

        return weights_to_parameters(weights)
