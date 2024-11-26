import torch
from torch import tensor as tt
from torch.distributions import Categorical
import random
import time
from collections import namedtuple, deque
from torch import nn as nn
from torch.nn.functional import one_hot as one_hot  # one_hot(tensor, num_classes)
import torch.nn.functional as F
import numpy as np
import itertools
import copy
import utils
import os

if torch.cuda.is_available():
    print("Using GPU!")
    device = torch.device("cuda")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    print("Using CPU!")
    device = torch.device("cpu")

class DNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, hidden_layers):
        super(DNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.layers = nn.ModuleList()
        self.dimensions = [input_size] + [hidden_size]*hidden_layers + [output_size]
        self.layers.append(nn.Linear(self.input_size, self.hidden_size))
        for i in range(self.hidden_layers):
            self.layers.append(nn.Linear(self.hidden_size, self.hidden_size))
        self.layers.append(nn.Linear(self.hidden_size, self.output_size))

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = F.relu(x)
        return x
