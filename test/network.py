# Copyright (c) 2021: Zhiyuan Nan (namjw@hanyang.ac.kr).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import os
import numpy as np
import math
import random
from test.utils import build_mlp, initialize_weight

class QNetwork(nn.Module):
    def __init__(self, n_states, n_actions, args):
        super(QNetwork, self).__init__()
        self.args = args
        self.device = args.device
        if not args.use_noisy_layer:
            self.net = build_mlp(
                input_dim=n_states,
                output_dim=n_actions,
                hidden_units=args.hidden_units,
                hidden_activation=nn.ReLU(),
                ).apply(initialize_weight)
        else:
            self.feature = nn.Linear(n_states, args.hidden_units[0])
            self.noisy_layer1 = NoisyLinear(args.hidden_units[0], args.hidden_units[1])
            self.noisy_layer2 = NoisyLinear(args.hidden_units[1], n_actions)

            self.apply(initialize_weight)

        self.to(self.device)

    def forward(self, state):
        if not self.args.use_noisy_layer:
            out = self.net(state)
        else:
            feature = F.relu(self.feature(state))
            hidden = F.relu(self.noisy_layer1(feature))
            out = self.noisy_layer2(hidden)
        return out

    def reset_noise(self):
        self.noisy_layer1.reset_noise()
        self.noisy_layer2.reset_noise()

class DuelingNetwork(nn.Module):
    def __init__(self, n_states, n_actions, args):
        super(DuelingNetwork, self).__init__()
        self.device = args.device

        self.feature = build_mlp(
                input_dim=n_states,
                output_dim=args.hidden_units[0],
                hidden_units=args.hidden_units[:1],
                hidden_activation=nn.ReLU(),
                output_activation=nn.ReLU(),
                ).apply(initialize_weight)

        self.advantage = build_mlp(
                input_dim=args.hidden_units[0],
                output_dim=n_actions,
                hidden_units=args.hidden_units[1:],
                hidden_activation=nn.ReLU(),
                ).apply(initialize_weight)

        self.value = build_mlp(
                input_dim=args.hidden_units[0],
                output_dim=1,
                hidden_units=args.hidden_units[1:],
                hidden_activation=nn.ReLU(),
                ).apply(initialize_weight)

        self.to(self.device)

    def forward(self, state):
        feature = self.feature(state)
        advantage = self.advantage(feature)
        value = self.value(feature)

        # Here we calculate advantage Q(s,a) = A(s,a) + V(s)
        out = value + advantage - advantage.mean(dim=-1, keepdim=True)

        return out

class DuelingTwinNetwork(nn.Module):
    def __init__(self, n_states, n_actions, args):
        super(DuelingTwinNetwork, self).__init__()
        self.device = args.device

        self.feature = build_mlp(
                input_dim=n_states,
                output_dim=args.hidden_units[0],
                hidden_units=args.hidden_units[:1],
                hidden_activation=nn.ReLU(),
                output_activation=nn.ReLU(),
                ).apply(initialize_weight)

        self.advantage1 = build_mlp(
                input_dim=args.hidden_units[0],
                output_dim=n_actions,
                hidden_units=args.hidden_units[1:],
                hidden_activation=nn.ReLU(),
                ).apply(initialize_weight)

        self.value1 = build_mlp(
                input_dim=args.hidden_units[0],
                output_dim=1,
                hidden_units=args.hidden_units[1:],
                hidden_activation=nn.ReLU(),
                ).apply(initialize_weight)

        self.advantage2 = build_mlp(
                input_dim=args.hidden_units[0],
                output_dim=n_actions,
                hidden_units=args.hidden_units[1:],
                hidden_activation=nn.ReLU(),
                ).apply(initialize_weight)

        self.value2 = build_mlp(
                input_dim=args.hidden_units[0],
                output_dim=1,
                hidden_units=args.hidden_units[1:],
                hidden_activation=nn.ReLU(),
                ).apply(initialize_weight)

        self.to(self.device)

    def forward(self, state):
        feature = self.feature(state)
        advantage1 = self.advantage1(feature)
        value1 = self.value1(feature)

        # Here we calculate advantage Q(s,a) = A(s,a) + V(s)
        out = value1 + advantage1 - advantage1.mean(dim=-1, keepdim=True)

        return out

    def get_double_q(self, state):
        feature = self.feature(state)

        advantage1 = self.advantage1(feature)
        value1 = self.value1(feature)

        advantage2 = self.advantage2(feature)
        value2 = self.value2(feature)

        q1 = value1 + advantage1 - advantage1.mean(dim=-1, keepdim=True)
        q2 = value2 + advantage2 - advantage2.mean(dim=-1, keepdim=True)

        return q1, q2

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(T.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(T.Tensor(out_features, in_features))
        self.register_buffer("weight_epsilon", T.Tensor(out_features, in_features))

        self.bias_mu = nn.Parameter(T.Tensor(out_features))
        self.bias_sigma = nn.Parameter(T.Tensor(out_features))
        self.register_buffer("bias_epsilon", T.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        return F.linear(x, self.weight_mu+self.weight_sigma*self.weight_epsilon, self.bias_mu+self.bias_sigma*self.bias_epsilon)

    @staticmethod
    def scale_noise(size):
        x = T.randn(size)
        return x.sign().mul(x.abs().sqrt())

class DeterministicPolicy(nn.Module): # Deterministic Policy Gradient(DPG), Deep Deterministic Policy Gradient(DDPG), Twin Delayed Deep Deterministic Policy Gradients(TD3)
    def __init__(self, n_states, n_actions, args, max_action=None):
        super(DeterministicPolicy, self).__init__()
        self.device = args.device
        self.max_action = max_action

        self.pi = build_mlp(
                input_dim=n_states,
                output_dim=n_actions,
                hidden_units=args.hidden_units,
                hidden_activation=nn.ReLU(),
                output_activation=nn.Tanh(),
                ).apply(initialize_weight)

        self.to(self.device)

    def forward(self, state):
        if self.max_action == None: return self.pi(state)      # action -> tanh() -> [-1,1]
        return self.max_action * self.pi(state)                # max_action -> [-max_action, max_action]

class ActorA2C(nn.Module): # Advantage Actor-Critic
    def __init__(self, n_states, n_actions, args):
        super(ActorA2C, self).__init__()
        self.args = args
        self.device = args.device

        self.net = build_mlp(
                input_dim=n_states,
                output_dim=2*n_actions,
                hidden_units=args.hidden_units,
                hidden_activation=nn.ReLU(),
                ).apply(initialize_weight)

        self.to(self.device)

    def forward(self, state):
        mu, log_std = T.chunk(self.net(state), 2, dim=-1)
        mu = T.tanh(mu) * 2
        log_std = F.softplus(log_std)
        std = T.exp(log_std)

        return mu, std

class ActorPPO(nn.Module): # Proximal Policy Optimization
    def __init__(self, n_states, n_actions, args):
        super(ActorPPO, self).__init__()
        self.args = args
        self.device = args.device

        self.mu = build_mlp(
                input_dim=n_states,
                output_dim=n_actions,
                hidden_units=args.hidden_units,
                hidden_activation=nn.ReLU(),
                ).apply(initialize_weight)

        self.log_std = nn.Parameter(T.zeros((1, n_actions)) -0.5, requires_grad=True)

        self.to(self.device)

    def forward(self, state):
        mu = self.mu(state)
        std = T.exp(self.log_std).expand_as(mu)

        return mu, std

class ActorSAC(nn.Module): # Soft Actor-Critic
    def __init__(self, n_states, n_actions, args, max_action=None, min_log_std=-20, max_log_std=2):
        super(ActorSAC, self).__init__()
        self.args = args
        self.device = args.device
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.max_action = max_action

        self.net = build_mlp(
                input_dim=n_states,
                output_dim=2*n_actions,
                hidden_units=args.hidden_units,
                hidden_activation=nn.ReLU(),
                ).apply(initialize_weight)

        self.to(self.device)

    def forward(self, state):
        mu, log_std = T.chunk(self.net(state), 2, dim=-1)
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

class CriticQ(nn.Module): # Action Value Function
    def __init__(self, n_states, n_actions, args):
        super(CriticQ, self).__init__()
        self.device = args.device

        self.value = build_mlp(
                input_dim=n_states + n_actions,
                output_dim=1,
                hidden_units=args.hidden_units,
                hidden_activation=nn.ReLU(),
                ).apply(initialize_weight)

        self.to(self.device)

    def forward(self, state, action):
        cat = T.cat((state, action), dim=-1)
        return self.value(cat)

class CriticV(nn.Module): # State Value Function
    def __init__(self, n_states, args):
        super(CriticV, self).__init__()
        self.device = args.device

        self.value = build_mlp(
                input_dim=n_states,
                output_dim=1,
                hidden_units=args.hidden_units,
                hidden_activation=nn.ReLU(),
                ).apply(initialize_weight)

        self.to(self.device)

    def forward(self, state):
        return self.value(state)

class CriticTwin(nn.Module): # Twin Delayed Deep Deterministic Policy Gradients(TD3), Double Deep Q Network
    def __init__(self, n_states, n_actions, args):
        super(CriticTwin, self).__init__()
        self.device = args.device

        self.value1 = build_mlp(
                input_dim=n_states + n_actions,
                output_dim=1,
                hidden_units=args.hidden_units,
                hidden_activation=nn.ReLU(),
                ).apply(initialize_weight)

        self.value2 = build_mlp(
                input_dim=n_states + n_actions,
                output_dim=1,
                hidden_units=args.hidden_units,
                hidden_activation=nn.ReLU(),
                ).apply(initialize_weight)

        self.to(self.device)

    def forward(self, state, action):
        cat = T.cat((state, action), dim=-1)
        return self.value1(cat)

    def get_double_q(self, state, action):
        cat = T.cat((state, action), dim=-1)
        return self.value1(cat), self.value2(cat)