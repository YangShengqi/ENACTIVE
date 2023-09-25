import torch.nn as nn
import torch
import os
import time
from utils.math import *

log_protect = 1e-5
multinomial_protect = 1e-10

class DiscretePolicy(nn.Module):
    def __init__(self, state_dim, action_num, hidden_size=(128, 128), activation='tanh'):
        super().__init__()
        self.is_disc_action = True
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.affine_layers = nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.action_head = nn.Linear(last_dim, action_num)
        # self.action_head.weight.data.mul_(0.3)
        # self.action_head.bias.data.mul_(0.1)

        set_init(self.affine_layers)
        set_init([self.action_head])

    def forward(self, x):
        #temp = time.time()
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        x = self.action_head(x)
        action_prob = torch.softmax(x, dim=-1)
        return action_prob

    def select_action(self, x:list, mask:list):
        action_prob = self.forward(torch.FloatTensor(x).unsqueeze(0))
        if mask is not None:
            for index, m in enumerate(mask):
                if m == 0.0:
                    action_prob[-1][index] = 0.0
        action_prob = action_prob+multinomial_protect
        action = action_prob.multinomial(1)
        return action.tolist()[0]

    def get_kl(self, x):
        action_prob1 = self.forward(x,None)#TODO
        action_prob0 = action_prob1.detach()
        kl = action_prob0 * (torch.log(action_prob0) - torch.log(action_prob1))
        return kl.sum(1, keepdim=True)

    def get_log_prob(self, x, actions):
        action_prob = self.forward(x)
        action_prob = action_prob.gather(1, actions.long())
        action_prob =action_prob+log_protect
        return torch.log(action_prob)

    def get_fim(self, x):
        action_prob = self.forward(x,None)#TODO
        M = action_prob.pow(-1).view(-1).detach()
        return M, action_prob, {}

