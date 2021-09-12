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
        self.log_std_min = args.log_std_min
        self.log_std_max = args.log_std_max

        self.feature = nn.Sequential(nn.Linear(n_states, args.ac_hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(args.ac_hidden_size, args.ac_hidden_size),
                                    nn.ReLU(),
                                    )

        self.log_std = nn.Linear(args.ac_hidden_size, n_actions)
        self.mu = nn.Linear(args.ac_hidden_size, n_actions)
        self.to(self.device)

    def forward(self, state, evaluate=False, with_logprob=True):
        feature = self.feature(state)

        mu = self.mu(feature)
        log_std = self.log_std(feature)
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

class MultiCritic(nn.Module):
    def __init__(self, n_states, n_actions, args):
        super(MultiCritic, self).__init__()
        self.device = args.device
        self.nets = []
        self.n_quantiles = args.n_quantiles
        self.n_nets = args.n_nets

        for i in range(self.n_nets):
            net = nn.Sequential(nn.Linear(n_states + n_actions, args.cri_hidden_size),
                                nn.ReLU(),
                                nn.Linear(args.cri_hidden_size, args.cri_hidden_size),
                                nn.ReLU(),
                                nn.Linear(args.cri_hidden_size, args.cri_hidden_size),
                                nn.ReLU(),
                                nn.Linear(args.cri_hidden_size, self.n_quantiles),
                                )
            self.add_module(f'qf{i}', net)
            self.nets.append(net)
        self.to(self.device)

    def forward(self, state, action):
        cat = T.cat((state, action), dim=1)
        quantiles = T.stack(tuple(net(cat) for net in self.nets), dim=1)
        return quantiles
