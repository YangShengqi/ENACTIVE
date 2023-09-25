import torch.nn as nn
import torch
from utils.math import *


class CommNetWork(nn.Module):
    def __init__(self, input_dim, value_hidden_size,  n_agents, seq_len=1, init_method='xavier'):
        super(CommNetWork, self).__init__()
        self.rnn_hidden_dim = value_hidden_size
        self.n_agents = n_agents
        self.seq_len = seq_len
        self.GRU_layers_num = 1
        self.input_shape = input_dim

        self.encoding = nn.Linear(self.input_shape, self.rnn_hidden_dim[0])
        self.f_obs = nn.Linear(self.rnn_hidden_dim[0], self.rnn_hidden_dim[0])
        self.f_comm = nn.ModuleList()
        for _ in range(self.GRU_layers_num):
            self.f_comm.append(nn.GRUCell(self.rnn_hidden_dim[0], self.rnn_hidden_dim[0]))
        self.value_head = nn.Linear(self.rnn_hidden_dim[0], self.rnn_hidden_dim[1])
        self.decoding = nn.Linear(self.rnn_hidden_dim[1], self.rnn_hidden_dim[2])

        set_init([self.encoding], method=init_method)
        set_init([self.f_obs], method=init_method)
        #set_init(self.f_comm, method=init_method)
        set_init([self.decoding], method=init_method)

    def forward(self, obs):
        obs_encoding = torch.relu(self.encoding(obs))
        h_out = self.f_obs(obs_encoding)     # 1 2 4 128

        h = h_out
        for k in range(self.GRU_layers_num):
            h = h.reshape(-1, self.n_agents, self.rnn_hidden_dim[0])
            c = h.reshape(-1, 1, self.n_agents * self.rnn_hidden_dim[0])
            c = c.repeat(1, self.n_agents, 1)
            mask = (1 - torch.eye(self.n_agents))
            mask = mask.view(-1, 1).repeat(1, self.rnn_hidden_dim[0]).view(self.n_agents, -1)
            mask = mask.to(c.device)
            c = c * mask.unsqueeze(0)
            c = c.reshape(-1, self.n_agents, self.n_agents, self.rnn_hidden_dim[0])
            c = c.mean(dim=-2)  #todo? mean()?
            h = h.reshape(-1, self.rnn_hidden_dim[0])
            c = c.reshape(-1, self.rnn_hidden_dim[0])

            h = self.f_comm[k](c, h)

        h = self.value_head(h)
        out = self.decoding(h).view(-1, self.n_agents, self.seq_len, self.rnn_hidden_dim[2]).squeeze(2)
        return out