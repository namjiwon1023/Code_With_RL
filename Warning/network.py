import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy
import math
from utils import reset_parameters, reset_single_layer_parameters

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

