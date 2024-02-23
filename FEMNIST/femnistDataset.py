# Copyright 2021 Adap GmbH. All Rights Reserved.
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
"""Creates a PyTorch Dataset for Leaf Shakespeare."""
import pickle
from pathlib import Path
from typing import List

import numpy as np
from flwr.dataset.utils.common import XY
from torch.utils.data import Dataset




class FemnistDataset(Dataset[XY]):  # type: ignore
    """Creates a PyTorch Dataset for Leaf Shakespeare.

    Args:
        Dataset (torch.utils.data.Dataset): PyTorch Dataset
    """

    def __init__(self, path_to_pickle, transform=None):
        self.X = []
        self.Y =[]
        self.index = []
        self.tag = []
        self.transform = transform
        data = {}  # Create an empty dictionary
        for path in path_to_pickle:
            with open(path, 'rb') as open_file:
                data=pickle.load(open_file)  # Update contents of file1 to the dictionary
                self.X = self.X + data["x"]
                self.Y = self.Y + data["y"]
                self.index.append(data["idx"])
                self.tag.append(data["tag"])

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> XY:

        x = self.X[idx]
        y = self.Y[idx]

        if self.transform:
            x = self.transform(x)

        return x, y


