# Copyright (c) 2021: Zhiyuan Nan (namjw@hanyang.ac.kr).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import torch as T
import torch.nn as nn
import torch.optim as optim
import os
import random
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np


class CriticNetwork(nn.Module):
    def __init__(self, n_states, n_actions, args):
        super(CriticNetwork, self).__init__()
        self.device = args.device
        self.checkpoint = os.path.join(args.save_dir + '/' + args.env_name, 'SAC_critic.pth')

        self.critic1 = nn.Sequential(nn.Linear(n_states + n_actions, args.hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(args.hidden_size, args.hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(args.hidden_size, 1)
                                    )

        self.critic2 = nn.Sequential(nn.Linear(n_states + n_actions, args.hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(args.hidden_size, args.hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(args.hidden_size, 1)
                                    )

        self.loss_func = nn.MSELoss()

        self.reset_parameters(self.critic1)
        self.reset_parameters(self.critic2)
        self.optimizer = optim.Adam(self.parameters(), lr=args.critic_lr)

        self.to(self.device)

    def forward(self, state, action):
        cat = T.cat((state, action), dim=-1)
        Q1 = self.critic1(cat)
        Q2 = self.critic2(cat)
        return Q1, Q2


    def reset_parameters(self, Sequential, std=1.0, bias_const=1e-6):
        for layer in Sequential:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, std)
                nn.init.constant_(layer.bias, bias_const)

    def save_model(self):
        T.save(self.state_dict(), self.checkpoint)

    def load_model(self):
        self.load_state_dict(T.load(self.checkpoint))
