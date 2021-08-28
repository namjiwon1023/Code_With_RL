# Copyright (c) 2021: Zhiyuan Nan (namjw@hanyang.ac.kr).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import torch as T
import torch.nn as nn
from torch.distributions import Normal
from network.AutoEncoder import Encoder
from utils.utils import weight_init

class ActorSAC(nn.Module):
    def __init__(self, obs_shape, action_shape, feature_dim, device):
        super(ActorSAC, self).__init__()
        self.encoder = Encoder(obs_shape, feature_dim, device)
        self.log_std_min = -10
        self.log_std_max = 2

        # feature_dim : encoder -> feature size
        self.net = nn.Sequential(
                    nn.Linear(feature_dim, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 2 * action_shape[0])
                    )

        self.apply(weight_init)
        self.to(device)

    def forward(self, obs, evaluate=False, with_logprob=True, detach_encoder=False):
        obs = self.encoder(obs, detach=detach_encoder)

        mu, log_std = self.net(obs).chunk(2, dim=-1)

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

        return action, log_prob, log_std

class QFunction(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
            )

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)
        cat = T.cat([obs, action], dim=1)
        return self.net(cat)

class Critic(nn.Module):
    def __init__(self, obs_shape, action_shape, feature_dim, device):
        super(Critic, self).__init__()

        self.encoder = Encoder(obs_shape, feature_dim, device)

        self.Q1 = QFunction(feature_dim, action_shape[0])
        self.Q2 = QFunction(feature_dim, action_shape[0])

        self.apply(weight_init)
        self.to(device)

    def forward(self, obs, action, detach_encoder=False):
        obs = self.encoder(obs, detach=detach_encoder)

        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)

        return q1, q2
