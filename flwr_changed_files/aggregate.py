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
"""Aggregation functions for strategy implementations."""


from functools import reduce
from typing import List, Optional, Tuple, Dict

import numpy as np

from flwr.common import Weights


def aggregate(results: List[Tuple[Weights, int]]) -> Weights:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    # Compute average weights of each layer
    weights_prime: Weights = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime


def weighted_loss_avg(results: List[Tuple[int, float, Optional[float]]]) -> float:
    """Aggregate evaluation results obtained from multiple clients."""
    num_total_evaluation_examples = sum(
        [num_examples for num_examples, _, _ in results]
    )
    weighted_losses = [num_examples * loss for num_examples, loss, _ in results]
    return sum(weighted_losses) / num_total_evaluation_examples


def aggregate_qffl(
    weights: Weights, deltas: List[Weights], hs_fll: List[Weights]
) -> Weights:
    """Compute weighted average based on  Q-FFL paper."""
    demominator = np.sum(np.asarray(hs_fll))
    scaled_deltas = []
    for client_delta in deltas:
        scaled_deltas.append([layer * 1.0 / demominator for layer in client_delta])
    updates = []
    for i in range(len(deltas[0])):
        tmp = scaled_deltas[0][i]
        for j in range(1, len(deltas)):
            tmp += scaled_deltas[j][i]
        updates.append(tmp)
    new_weights = [(u - v) * 1.0 for u, v in zip(weights, updates)]
    return new_weights

def aggregate_drop(results: List[Tuple[Weights, int, str]], dropWeights: Dict[str, List], origWeights: Weights) -> Weights:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples, _ in results])
    total_examples_wDrop = []
    for i in range(len(origWeights)):
        total_examples_wDrop.append(np.full(origWeights[i].shape, num_examples_total))


    #dropWeights = [ (row:[1,4,5], col[5,8]), ... , (row:[1,4,5], col[5,8]), (row:[1,4,5], col[5,8]) ]
    # transform the list of weights into original format (0 if there's nothing)

    transformedResults = []
    for (clientWeights, num_examples, cid) in results:
        layer = 0
        transformedWeights = clientWeights

        if cid not in dropWeights:
            transformedResults.append((transformedWeights, num_examples))
            continue

        for [row, col] in dropWeights[cid]:
            transformedWeights[layer] = clientWeights[layer]

            colLen = len(col)
            rowLen = len(row)

            if (rowLen!=0):
                transformedWeights[layer] = np.insert(transformedWeights[layer], row - np.arange(len(row)), 0,axis=0)
                total_examples_wDrop[layer][np.ix_(row)] -= num_examples

            if (colLen!=0):
                transformedWeights[layer] = np.insert(transformedWeights[layer], col - np.arange(len(col)), 0, axis=1)
                total_examples_wDrop[layer][:,col] -= num_examples

                total_examples_wDrop[layer][np.ix_(row,col)] += num_examples
            layer += 1

        transformedResults.append((transformedWeights, num_examples))
          
    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in transformedResults
    ]

    # Compute average weights of each layer
    weights_prime: Weights = [
        np.divide(reduce(np.add, layer_updates), total_examples_wDrop[i])
        for i, layer_updates in enumerate(zip(*weighted_weights))
    ]

    return weights_prime
