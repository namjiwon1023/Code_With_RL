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

class MultiCriticTwin(nn.Module):
    def __init__(self, n_states, n_actions, args):
        super(MultiCriticTwin, self).__init__()
        self.device = args.device
        self.z_nets_1 = []
        self.z_nets_2 = []
        self.n_quantiles = args.n_quantiles
        self.n_nets = args.n_nets

        for i in range(self.n_nets):
            net_1 = nn.Sequential(
                                nn.Linear(n_states + n_actions, args.cri_doubleQ_hidden_size),
                                nn.ReLU(),
                                nn.Linear(args.cri_doubleQ_hidden_size, args.cri_doubleQ_hidden_size),
                                nn.ReLU(),
                                nn.Linear(args.cri_doubleQ_hidden_size, args.cri_doubleQ_hidden_size),
                                nn.ReLU(),
                                nn.Linear(args.cri_doubleQ_hidden_size, self.n_quantiles)
                                ).to(self.device)
            self.add_module(f'qf{i}', net_1)
            self.z_nets_1.append(net_1)

        for i in range(self.n_nets):
            net_2 = nn.Sequential(
                                nn.Linear(n_states + n_actions, args.cri_doubleQ_hidden_size),
                                nn.ReLU(),
                                nn.Linear(args.cri_doubleQ_hidden_size, args.cri_doubleQ_hidden_size),
                                nn.ReLU(),
                                nn.Linear(args.cri_doubleQ_hidden_size, args.cri_doubleQ_hidden_size),
                                nn.ReLU(),
                                nn.Linear(args.cri_doubleQ_hidden_size, self.n_quantiles)
                                ).to(self.device)
            self.add_module(f'qf{i}', net_2)
            self.z_nets_2.append(net_2)

        self.to(self.device)

    def forward(self, state, action):
        cat = T.cat((state, action), dim=1)
        quantiles_1 = T.stack(tuple(net1(cat) for net1 in self.z_nets_1), dim=1)
        quantiles_2 = T.stack(tuple(net2(cat) for net2 in self.z_nets_2), dim=1)
        return quantiles_1, quantiles_2

class CriticTwin(nn.Module):
    def __init__(self, n_states, n_actions, args):
        super(CriticTwin, self).__init__()
        self.device = args.device

        self.Q1 = nn.Sequential(nn.Linear(n_states + n_actions, args.cri_doubleQ_hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(args.cri_doubleQ_hidden_size, args.cri_doubleQ_hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(args.cri_doubleQ_hidden_size, 1)
                                    )

        self.Q2 = nn.Sequential(nn.Linear(n_states + n_actions, args.cri_doubleQ_hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(args.cri_doubleQ_hidden_size, args.cri_doubleQ_hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(args.cri_doubleQ_hidden_size, 1)
                                    )

        self.to(self.device)

    def forward(self, state, action):
        cat = T.cat((state, action), dim=-1)
        Q1 = self.Q1(cat)
        Q2 = self.Q2(cat)
        return Q1, Q2