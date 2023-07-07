"""
Adapted from evolutionary code written by github user Sir-Batman
https://github.com/AADILab/PyTorch-Evo-Strategies

Structure and main functions for basic single layer linear Neural Network
"""
import random

import torch
from torch import nn
import numpy as np
from os import getcwd, path


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hid_size, out_size):
        super(NeuralNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hid_size),
            nn.ReLU(inplace=True),
            nn.Linear(hid_size, out_size),
            nn.Sigmoid(),
        )
        self.model.requires_grad_(False)

    def run(self, x):
        return self.model(x)

    def get_weights(self):
        d = self.model.state_dict()
        return [d['0.weight'], d['2.weight']]

    def get_biases(self):
        d = self.model.state_dict()
        return [d['0.bias'], d['2.bias']]

    def get_model(self):
        return self.model.state_dict()

    def set_weights(self, weights):
        d = self.model.state_dict()
        d['0.weight'] = weights[0]
        d['2.weight'] = weights[1]
        self.model.load_state_dict(d)

    def set_biases(self, biases):
        d = self.model.state_dict()
        d['0.bias'] = biases[0]
        d['2.bias'] = biases[1]
        self.model.load_state_dict(d)

    def forward(self, x):
        x = torch.Tensor(x)
        flat_x = torch.flatten(x)
        logits = self.model(flat_x)
        return logits



