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


class FedDropShake(Strategy):
    """Configurable FedAvg strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        fraction_fit: float = 0.1,
        fraction_eval: float = 0.1,
        min_fit_clients: int = 2,
        min_eval_clients: int = 2,
        min_available_clients: int = 2,
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
        self.unchagedWeights = [[] for x in range(11)]
        self.defDropWeights = [[] for x in range(11)]
        self.prevDropWeights = [[] for x in range(11)]
        self.changeThreshold = 30
        self.changeIncrement = 1.0
        self.roundCounter = 0
        self.stopChange = False
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
                print(p_val)
                config_drop = self.on_fit_config_fn(rnd,p_val)

                # CHANGE HERE for each dropout method. See method for clarification on each input to the method
                # drop_rand - random dropout
                # drop_order - ordered dropout
                # drop_dynamic - invariant dropout
                fit_ins_drop = FitIns(self.drop_dynamic(parameters, p_val, [1,5], 2, client.cid), config_drop)
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

        if (rnd > 25 and self.stopChange != True):
            self.roundCounter += 1
            if ( self.roundCounter >= 5): 
                self.changeThreshold += self.changeIncrement
                self.roundCounter = 0
                print("threshold updated to: ", self.changeThreshold)
        
        # find the invariant neurons
        self.find_stable(self.parameters, weights_results, [1,5], 2)

        # find minimum weight changes to initialize threshold (only done once)
        self.find_min(self.parameters, weights_results, [1,5], rnd)
        aggregated_weights = aggregate_drop(weights_results, self.droppedWeights, parameters_to_weights(self.initial_parameters))
 
        def time(elem):
                return elem[1].fit_duration

        results.sort(key=time)

        # set number of stragglers 
        numStrag = int(len(results) * 0.2)
        if (numStrag < 1):
                numStrag = 1

        if (len(self.straggler) == 0 and rnd > 1): 
            for i in range(numStrag):
                numInClass = int(numStrag * 0.25)
                newStrag = results[len(results) - 1 - i]
                self.straggler[newStrag[0].cid] = newStrag[1].fit_duration

                # set the p value for the new straggler (constant sub-model size)
                self.p_val[newStrag[0].cid] = self.constant_pval
                #if (i < numInClass):
                #    self.p_val[newStrag[0].cid] = 0.65
                #elif (i < (2* numInClass + 1) ):
                #    self.p_val[newStrag[0].cid] = 0.75
                #elif (i < (3* numInClass + 2)):
                #    self.p_val[newStrag[0].cid] = 0.85
                #else:
                #    self.p_val[newStrag[0].cid] = 0.95

            self.straggler = dict(sorted(self.straggler.items(), key=lambda item: item[1]))
            #print(self.straggler)
            #print(self.p_val)

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
            numInClass = int(numStrag * 0.25)
            for i in range(numStrag):

                # set the p value for the new straggler (constant sub-model size)
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
        print(len(weights))
        if cid not in self.droppedWeights:
            self.droppedWeights[cid] = [[[],[]] for x in range(len(weights))]

        for idx in idxList:
            shape = weights[idx+1].shape
            list = random.sample(range(1, shape[1]), shape[1] - int(p * shape[1]))
            list.sort()

            listExt = []
            size = 128
            for drop in list:
                listExt.append(drop + 0)
                listExt.append(drop + 128)
                listExt.append(drop + 256)
                listExt.append(drop + 384)
            listExt.sort()

            self.droppedWeights[cid][idx][0] = listExt.copy()
            self.droppedWeights[cid][idx + 1][0] = listExt.copy()
            self.droppedWeights[cid][idx + 1][1] = list.copy()
            self.droppedWeights[cid][idx + 2][0] = listExt.copy()
            self.droppedWeights[cid][idx + 3][0] = listExt.copy()
            self.droppedWeights[cid][idx + 4][1] = list.copy()

            self.prevDropWeights[idx] = list.copy()
            print("Dropped weights idx ", idx, ": ", (self.prevDropWeights[idx]))

            # remove each row/column from the back
            print(listExt)
            weights[idx] = np.delete(weights[idx], listExt, 0)
            weights[idx + 1] = np.delete(weights[idx + 1], listExt, 0)
            weights[idx + 1] = np.delete(weights[idx + 1], list, 1)
            weights[idx + 2] = np.delete(weights[idx + 2], listExt, 0)
            weights[idx + 3] = np.delete(weights[idx + 3], listExt, 0)
            weights[idx + 4] = np.delete(weights[idx + 4], list, 1)

        return weights_to_parameters(weights)

    #drop individual neuron or filter
    # if we're doing a reduction of filters at the last covolutioanal layer ( we need to delete 7*7 colmns in first fully connected matrix for each filter)


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
            shape = weights[idx+1].shape
            numToDrop = shape[1] - int(p * shape[1])
            list = [x for x in range(shape[1] - numToDrop, shape[1])]

            listExt = []
            size = 128
            for drop in list:
                listExt.append(drop + 0)
                listExt.append(drop + 128)
                listExt.append(drop + 256)
                listExt.append(drop + 384)
            listExt.sort()

            self.droppedWeights[cid][idx][0] = listExt.copy()
            self.droppedWeights[cid][idx + 1][0] = listExt.copy()
            self.droppedWeights[cid][idx + 1][1] = list.copy()
            self.droppedWeights[cid][idx + 2][0] = listExt.copy()
            self.droppedWeights[cid][idx + 3][0] = listExt.copy()
            self.droppedWeights[cid][idx + 4][1] = list.copy()

            self.prevDropWeights[idx] = list.copy()
            print("Dropped weights idx ", idx, ": ", (self.prevDropWeights[idx]))

            # remove each row/column from the back
            weights[idx] = np.delete(weights[idx], listExt, 0)
            weights[idx + 1] = np.delete(weights[idx + 1], listExt, 0)
            weights[idx + 1] = np.delete(weights[idx + 1], list, 1)
            weights[idx + 2] = np.delete(weights[idx + 2], listExt, 0)
            weights[idx + 3] = np.delete(weights[idx + 3], listExt, 0)
            weights[idx + 4] = np.delete(weights[idx + 4], list, 1)

        return weights_to_parameters(weights)

    # find invariant neurons
    def find_stable(self, parameters: Parameters, results: List[Tuple[Weights, int, str]], idxList: List[int], idxConvFC: int):

        weights = parameters_to_weights(parameters)

        list = []

        difference = []
        for i in range(len(weights)):

            difference.append(np.full(weights[i].shape, 0))

        for cWeights, num_examples, cid in results:
            if cid in self.straggler:
                continue
            clientWeights = cWeights

            for i in range(len(weights)):
                difference[i] += (np.abs(clientWeights[i] - weights[i]) <= np.abs(self.changeThreshold* weights[i])) * 1
               
        for i in range(len(difference)):
            difference[i] = difference[i] > (0.75 * (len(results) - len(self.straggler)))


        list = [[] for x in range(len(weights))]


        for idx in idxList:
            shape = weights[idx].shape
            dim = [x for x in range(weights[idx].ndim)]
            dim.remove(1)

            # perform reduction for all other dimensions (so we know which idx has constant weights
            idx11Layer =  np.all(difference[idx+1], axis=tuple(dim))

            idx4Layer = np.all(difference[idx+4], axis=tuple(dim))

            dim = [x for x in range(weights[idx].ndim)]
            dim.remove(0)
            idx0Layer = np.all(difference[idx], axis=tuple(dim))
            idx10Layer = np.all(difference[idx+1], axis=tuple(dim))
            idx2Layer = difference[idx+2]
            idx3Layer = difference[idx+3]
            reduced0List = np.array([])
            reduced10List = np.array([])
            reduced2List = np.array([])
            reduced3List = np.array([])
            for i in range(len(idx11Layer)):
                reduced0List = np.array(np.append(reduced0List, (idx0Layer[i] and idx0Layer[i+128] and idx0Layer[i+256] and idx0Layer[i+384])), dtype=bool)
                reduced10List = np.array(np.append(reduced10List, (idx10Layer[i] and idx10Layer[i+128] and idx10Layer[i+256] and idx10Layer[i+384])), dtype=bool)
                reduced2List = np.array(np.append(reduced2List, (idx0Layer[i] and idx2Layer[i+128] and idx2Layer[i+256] and idx2Layer[i+384])), dtype=bool)
                reduced3List = np.array(np.append(reduced3List, (idx0Layer[i] and idx3Layer[i+128] and idx3Layer[i+256] and idx3Layer[i+384])), dtype=bool)

            noChangeIdx =  idx11Layer & idx4Layer & reduced0List & reduced10List & reduced2List & reduced3List
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

        self.unchagedWeights = list

        return list


    # initialize threshold value (find minimum change in weights)
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
        
        minThresh = 0.0
        for idx in idxList:
            shape = weights[idx].shape
            dim = [x for x in range(weights[idx].ndim)]
            dim.remove(1)

            # perform reduction for all other dimensions (so we know which idx has constant weights
            idx11Layer =  np.amax(difference[idx+1], axis=tuple(dim))

            idx4Layer = np.amax(difference[idx+4], axis=tuple(dim))

            dim = [x for x in range(weights[idx].ndim)]
            dim.remove(0)

            idx0Layer = np.amax(difference[idx], axis=tuple(dim))
            idx10Layer = np.amax(difference[idx+1], axis=tuple(dim))
            idx2Layer = difference[idx+2]
            idx3Layer = difference[idx+3]

            reduced0List = np.array([])
            reduced10List = np.array([])
            reduced2List = np.array([])
            reduced3List = np.array([])
            for i in range(len(idx11Layer)):
                a = np.array([idx0Layer[i], idx0Layer[i+128], idx0Layer[i+256], idx0Layer[i+384]])
                reduced0List = np.array(np.append(reduced0List, np.amax(a)), dtype=float)
                b = np.array([idx10Layer[i], idx10Layer[i+128], idx10Layer[i+256], idx10Layer[i+384]])
                reduced10List = np.array(np.append(reduced10List, np.amax(b)), dtype=float)
                c = np.array([idx2Layer[i], idx2Layer[i+128], idx2Layer[i+256], idx2Layer[i+384]])
                reduced2List = np.array(np.append(reduced2List, np.amax(c)), dtype=float)
                d = np.array([idx10Layer[i], idx3Layer[i+128], idx3Layer[i+256], idx3Layer[i+384]])
                reduced3List = np.array(np.append(reduced3List, np.amax(d)), dtype=float)
            a = np.maximum(idx11Layer, idx4Layer)
            b = np.maximum(np.maximum(np.maximum(reduced0List, reduced10List),reduced2List), reduced3List)



            sum = np.maximum(a,b)

            noChangeIdx =  np.argsort(sum)
          
            print("% difference: ", sum[noChangeIdx[0]])
            if (rnd ==2 or rnd == 3):
                minThresh += sum[noChangeIdx[0]]

        if (rnd == 2):
            self.changeThreshold = (minThresh/ len(idxList))
            print("threshold updated to: ", self.changeThreshold)
        if (rnd == 3):
            self.changeThreshold = (self.changeThreshold + (minThresh/ len(idxList))) /2
            print("threshold updated to: ", self.changeThreshold)


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
            second_0 = 0
            second_1 = 1
            third = 0
            fourth = 0
            fifth = 1

            shape = weights[idx +1].shape
            numToDrop = shape[second_1 ] - int(p * shape[second_1])
            if len(self.unchagedWeights[idx]) >= numToDrop:
                if (idx == 1 or idx ==5):
                    self.stopChange = True

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
                fullList = [x for x in range(shape[second_1])]
                for x in self.unchagedWeights[idx]:
                    fullList.remove(x)
                list = random.sample(fullList, numToDrop - len(self.unchagedWeights[idx]))
                list.extend(self.unchagedWeights[idx])
                list.sort()

            listExt = []
            size = 128
            for drop in list:
                listExt.append(drop + 0)
                listExt.append(drop + 128)
                listExt.append(drop + 256)
                listExt.append(drop + 384)
            listExt.sort()

            self.droppedWeights[cid][idx][0] = listExt.copy()
            self.droppedWeights[cid][idx + 1][0] = listExt.copy()
            self.droppedWeights[cid][idx + 1][1] = list.copy()
            self.droppedWeights[cid][idx + 2][0] = listExt.copy()
            self.droppedWeights[cid][idx + 3][0] = listExt.copy()
            self.droppedWeights[cid][idx + 4][1] = list.copy()

            self.prevDropWeights[idx] = list.copy()
            print("Dropped weights idx ", idx, ": ", (self.prevDropWeights[idx]))

            # remove each row/column from the back
            weights[idx] = np.delete(weights[idx], listExt, 0)
            weights[idx + 1] = np.delete(weights[idx + 1], listExt, 0)
            weights[idx + 1] = np.delete(weights[idx + 1], list, 1)
            weights[idx + 2] = np.delete(weights[idx + 2], listExt, 0)
            weights[idx + 3] = np.delete(weights[idx + 3], listExt, 0)
            weights[idx + 4] = np.delete(weights[idx + 4], list, 1)

        return weights_to_parameters(weights)


    