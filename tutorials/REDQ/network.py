import torch as T
import torch.nn as nn
import os
import numpy as np
from torch.distributions import Normal

class Actor(nn.Module):
    def __init__(self, n_states, n_actions, args):
        super(Actor, self).__init__()
        self.args = args
        self.device = args.device
        self.log_std_min = -20
        self.log_std_max = 2

        self.feature = nn.Sequential(
                                nn.Linear(n_states, args.ac_hidden_size),
                                nn.ReLU(),
                                nn.Linear(args.ac_hidden_size, args.ac_hidden_size),
                                nn.ReLU(),
                                )
        self.log_std = nn.Linear(args.ac_hidden_size, n_actions)
        self.mu = nn.Linear(args.ac_hidden_size, n_actions)

        reset_parameters(self.feature)
        reset_single_layer_parameters(self.log_std)
        reset_single_layer_parameters(self.mu)

        self.to(self.device)

    def forward(self, state, evaluate=False, with_logprob=True):

        feature = self.feature(state)
        log_std = self.log_std(feature)
        mu = self.mu(feature)

        log_std = T.clamp(log_std, self.log_std_min, self.log_std_max)
        std = T.exp(log_std)

        dist = Normal(mu, std)
        z = dist.rsample()

        if evaluate:
            action = mu.tanh()
        else:
            action = z.tanh()

        if with_logprob:
            log_prob = dist.log_prob(z) - T.log(1 - action.pow(2) + 1e-7)
            log_prob = log_prob.sum(-1, keepdim=True)
        else:
            log_prob = None

        return action, log_prob

class QNet(nn.Module):
    def __init__(self, n_states, n_actions, args):
        super(QNet, self).__init__()
        self.device = args.device
        self.Q = nn.Sequential(
                            nn.Linear(n_states + n_actions, args.cri_hidden_size),
                            nn.ReLU(),
                            nn.Linear(args.cri_hidden_size, args.cri_hidden_size),
                            nn.ReLU(),
                            nn.Linear(args.cri_hidden_size, 1),
                            )
        reset_parameters(self.Q)

        self.to(self.device)

    def forward(self, state, action):
        cat = T.cat([state, action], dim=1)
        value = self.Q(cat)
        return value

def reset_parameters(Sequential, std=1.0, bias_const=1e-6):
    for layer in Sequential:
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, std)
            nn.init.constant_(layer.bias, bias_const)

def reset_single_layer_parameters(layer, std=1.0, bias_const=1e-6):
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
