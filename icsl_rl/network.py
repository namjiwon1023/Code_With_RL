# Copyright (c) 2021: Zhiyuan Nan (namjw@hanyang.ac.kr).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import math
from torch.distributions import Normal, Categorical
import random

class QNetwork(nn.Module):
    def __init__(self, n_states, n_actions, args):
        super(QNetwork, self).__init__()
        self.args = args
        self.device = args.device
        if not args.use_noisy_layer:
            self.critic = nn.Sequential(nn.Linear(n_states, args.hidden_size),
                                        nn.ReLU(),
                                        nn.Linear(args.hidden_size, args.hidden_size),
                                        nn.ReLU(),
                                        nn.Linear(args.hidden_size, n_actions)
                                        )
            reset_parameters(self.critic)
        else:
            self.feature = nn.Linear(n_states, args.hidden_size)
            self.noisy_layer1 = NoisyLinear(args.hidden_size, args.hidden_size)
            self.noisy_layer2 = NoisyLinear(args.hidden_size, n_actions)

            reset_single_layer_parameters(self.feature)

        self.to(self.device)

    def forward(self, state):
        if not self.args.use_noisy_layer:
            out = self.critic(state)
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

        self.feature = nn.Sequential(nn.Linear(n_states, args.hidden_size),
                                    nn.ReLU(),)

        self.advantage = nn.Sequential(
                                    nn.Linear(args.hidden_size, args.hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(args.hidden_size, n_actions),
                                        )


        self.value = nn.Sequential(
                                    nn.Linear(args.hidden_size, args.hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(args.hidden_size, 1),
                                        )

        reset_parameters(self.feature)
        reset_parameters(self.advantage)
        reset_parameters(self.value)

        self.to(self.device)

    def forward(self, state):
        feature = self.feature(state)
        advantage = self.advantage(feature)
        value = self.value(feature)

        # Here we calculate advantage Q(s,a) = A(s,a) + V(s)
        out = value + advantage - advantage.mean(dim=-1, keepdim=True)

        return out

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

class Actor(nn.Module): # Deterministic Policy Gradient(DPG), Deep Deterministic Policy Gradient(DDPG), Twin Delayed Deep Deterministic Policy Gradients(TD3)
    def __init__(self, n_states, n_actions, args, max_action=None):
        super(Actor, self).__init__()
        self.device = args.device
        self.max_action = max_action

        self.pi = nn.Sequential(nn.Linear(n_states, args.hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(args.hidden_size, args.hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(args.hidden_size, n_actions),
                                    nn.Tanh())

        reset_parameters(self.pi)

        self.to(self.device)

    def forward(self, state):
        u = self.pi(state)
        if self.max_action == None: return u
        return self.max_action*u

class ActorA2C(nn.Module): # Advantage Actor-Critic
    def __init__(self, n_states, n_actions, args):
        super(ActorA2C, self).__init__()
        self.args = args
        self.device = args.device

        self.feature = nn.Sequential(nn.Linear(n_states, args.hidden_size),
                                nn.ReLU(),
                                nn.Linear(args.hidden_size, args.hidden_size),
                                nn.ReLU(),
                                )

        self.mu = nn.Linear(args.hidden_size, n_actions)
        self.log_std = nn.Linear(args.hidden_size, n_actions)

        reset_parameters(self.feature)
        reset_single_layer_parameters(self.mu)
        reset_single_layer_parameters(self.log_std)

        self.to(self.device)

    def forward(self, state):
        feature = self.feature(state)
        mu = T.tanh(self.mu(feature)) * 2
        log_std = F.softplus(self.log_std(feature))
        std = T.exp(log_std)

        return mu, std

class ActorPPO(nn.Module): # Proximal Policy Optimization
    def __init__(self, n_states, n_actions, args):
        super(ActorPPO, self).__init__()
        self.args = args
        self.device = args.device

        self.mu = nn.Sequential(nn.Linear(n_states, args.hidden_size),
                                nn.ReLU(),
                                nn.Linear(args.hidden_size, args.hidden_size),
                                nn.ReLU(),
                                nn.Linear(args.hidden_size, n_actions),
                                )

        self.log_std = nn.Parameter(T.zeros((1, n_actions)) -0.5, requires_grad=True)

        reset_parameters(self.mu)

        self.to(self.device)

    def forward(self, state):

        mu = self.mu(state)

        std = T.exp(self.log_std).expand_as(mu)

        return mu, std

class ActorSAC(nn.Module): # Soft Actor-Critic
    def __init__(self, n_states, n_actions, args, max_action=None):
        super(ActorSAC, self).__init__()
        self.args = args
        self.device = args.device
        self.min_log_std = args.min_log_std
        self.max_log_std = args.max_log_std
        self.max_action = max_action

        self.feature = nn.Sequential(nn.Linear(n_states, args.hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(args.hidden_size, args.hidden_size),
                                    nn.ReLU(),
                                    )

        self.log_std = nn.Linear(args.hidden_size, n_actions)
        self.mu = nn.Linear(args.hidden_size, n_actions)

        reset_parameters(self.feature)
        reset_single_layer_parameters(self.log_std)
        reset_single_layer_parameters(self.mu)

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

class CriticQ(nn.Module): # Action Value Function
    def __init__(self, n_states, n_actions, args):
        super(CriticQ, self).__init__()
        self.device = args.device

        self.Value = nn.Sequential(nn.Linear(n_states + n_actions, args.hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(args.hidden_size, args.hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(args.hidden_size, 1)
                                    )

        reset_parameters(self.Value)

        self.to(self.device)

    def forward(self, state, action):
        cat = T.cat((state, action), dim=-1)
        Q = self.Value(cat)
        return Q

class CriticV(nn.Module): # State Value Function
    def __init__(self, n_states, args):
        super(CriticV, self).__init__()
        self.device = args.device

        self.Value = nn.Sequential(nn.Linear(n_states, args.hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(args.hidden_size, args.hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(args.hidden_size, 1)
                                    )

        reset_parameters(self.Value)

        self.to(self.device)

    def forward(self, state):
        V = self.Value(state)
        return V

class CriticTwin(nn.Module): # Twin Delayed Deep Deterministic Policy Gradients(TD3), Double Deep Q Network
    def __init__(self, n_states, n_actions, args):
        super(CriticTwin, self).__init__()
        self.device = args.device

        self.Value1 = nn.Sequential(nn.Linear(n_states + n_actions, args.hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(args.hidden_size, args.hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(args.hidden_size, 1)
                                    )

        self.Value2 = nn.Sequential(nn.Linear(n_states + n_actions, args.hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(args.hidden_size, args.hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(args.hidden_size, 1)
                                    )

        reset_parameters(self.Value1)
        reset_parameters(self.Value2)

        self.to(self.device)

    def forward(self, state, action):
        cat = T.cat((state, action), dim=-1)
        Q1 = self.Value1(cat)
        return Q1

    def get_double_q(self, state, action):
        cat = T.cat((state, action), dim=-1)
        Q1 = self.Value1(cat)
        Q2 = self.Value2(cat)
        return Q1, Q2

class DiagonalGaussianDistribution:

    def __init__(self, mu, log_std):
        self.mu = mu
        self.log_std = log_std

    def sample(self):
        return self.mu + T.exp(self.log_std) * T.randn_like(self.mu)

    def log_prob(self, value):
        return gaussian_likelihood(value, self.mu, self.log_std)

    def entropy(self):
        return 0.5 + 0.5 * np.log(2 * np.pi) + self.log_std.sum(axis=-1)

class MLPGaussianActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = T.nn.Parameter(T.as_tensor(log_std))
        self.mu_net = create_mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def forward(self, obs, act=None):
        mu = self.mu_net(obs)
        pi = DiagonalGaussianDistribution(mu, self.log_std)
        logp_a = None
        if act is not None:
            logp_a = pi.log_prob(act)
        return pi, logp_a

def reset_parameters(Sequential, std=1.0, bias_const=1e-6):
    for layer in Sequential:
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, std)
            nn.init.constant_(layer.bias, bias_const)

def reset_single_layer_parameters(layer, std=1.0, bias_const=1e-6):
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)

################# MLP building #################
def create_mlp(sizes, activation, output_activation=nn.Identity):
    # OpenAI style
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def build_mlp(
    input_dim,
    output_dim,
    hidden_units=[64, 64],
    hidden_activation=nn.Tanh(),
    output_activation=None,
):
    layers = []
    units = input_dim
    for next_units in hidden_units:
        layers.append(nn.Linear(units, next_units))
        layers.append(hidden_activation)
        units = next_units
    layers.append(nn.Linear(units, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)
########################################

def gaussian_likelihood(x, mu, log_std, eps=1e-8):
    pre_sum = -0.5 * (((x-mu)/(T.exp(log_std)+eps))**2 + 2*log_std + np.log(2*np.pi))
    return pre_sum.sum(axis=-1)

def conv2d_size_out(size, kernel_size, stride, padding):
    return ((size + 2 * padding - kernel_size) // stride) + 1

def initialize_weight(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
