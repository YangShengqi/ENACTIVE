import torch
import math
import torch.nn as nn
import numpy as np


def normal_entropy(std):
    var = std.pow(2)
    entropy = 0.5 + 0.5 * torch.log(2 * var * math.pi)
    return entropy.sum(1, keepdim=True)


def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)


def set_init(layers, method=None):
    for layer in layers:
        if method == "kaiming":
            nn.init.kaiming_normal_(layer.weight)  # use relu for activation
        elif method == "xavier":
            # nn.init.xavier_uniform_(layer.weight, gain=1)
            tanh_gain = torch.nn.init.calculate_gain("tanh")
            nn.init.xavier_normal_(layer.weight, gain=tanh_gain)
        elif method == "lstm_orthogonal_init":
            tanh_gain = torch.nn.init.calculate_gain("tanh")
            for name, param in layer.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param, gain=tanh_gain)
        else:
            # use origin method
            nn.init.normal_(layer.weight, mean=0., std=0.1)
            nn.init.constant_(layer.bias, 0.0)


def index_to_one_hot(index, dim):
    if isinstance(index, np.int) or isinstance(index, np.int64):
        one_hot = np.zeros(dim)
        one_hot[index] = 1.
    else:
        one_hot = np.zeros((len(index), dim))
        one_hot[np.arange(len(index)), index] = 1.
    return one_hot.squeeze().tolist()

