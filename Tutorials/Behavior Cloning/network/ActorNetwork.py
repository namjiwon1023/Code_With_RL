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

class ActorNetwork(nn.Module):
    def __init__(self, n_states, n_actions, args, max_action=None):
        super(ActorNetwork, self).__init__()
        self.args = args
        self.device = args.device
        self.min_log_std = args.min_log_std
        self.max_log_std = args.max_log_std
        self.max_action = max_action
        self.checkpoint = os.path.join(args.save_dir + '/' + args.env_name, 'SAC_actor.pth')

        self.feature = nn.Sequential(nn.Linear(n_states, args.hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(args.hidden_size, args.hidden_size),
                                    nn.ReLU(),
                                    )

        self.log_std = nn.Linear(args.hidden_size, n_actions)
        self.mu = nn.Linear(args.hidden_size, n_actions)

        self.reset_parameters(self.feature)
        self.single_layer_reset_parameters(self.log_std)
        self.single_layer_reset_parameters(self.mu)

        self.optimizer = optim.Adam(self.parameters(), lr=args.actor_lr)

        self.to(self.device)

    def forward(self, state):
        feature = self.feature(state)

        mu = self.mu(feature)
        log_std = self.log_std(feature)
        log_std = T.clamp(log_std, self.min_log_std, self.max_log_std)
        std = T.exp(log_std)

        dist = Normal(mu, std)
        z = dist.rsample()

        if self.args.evaluate:
            action = mu.tanh()
        else:
            action = z.tanh()

        if self.args.with_logprob:
            log_prob = dist.log_prob(z) - T.log(1 - action.pow(2) + 1e-7)
            log_prob = log_prob.sum(-1, keepdim=True)
        else:
            log_prob = None

        if self.max_action == None: return action, log_prob
        return self.max_action*action, log_prob


    def reset_parameters(self, Sequential, std=1.0, bias_const=1e-6):
        for layer in Sequential:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, std)
                nn.init.constant_(layer.bias, bias_const)

    def single_layer_reset_parameters(self, layer, std=1.0, bias_const=1e-6):
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, std)
                nn.init.constant_(layer.bias, bias_const)

    def save_model(self):
        T.save(self.state_dict(), self.checkpoint)

    def load_model(self):
        self.load_state_dict(T.load(self.checkpoint))
