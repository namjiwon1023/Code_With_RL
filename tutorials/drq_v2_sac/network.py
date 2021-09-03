import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import copy
import utils
from torch.distributions import Normal
import utils

class Encoder(nn.Module):
    def __init__(self, obs_shape, device):
        super().__init__()
        assert len(obs_shape) == 3   # C， W， H -> len(3, 84, 84) = 3

        self.output_logits = False
        self.device = device
        self.repr_dim = 32 * 35 * 35

        # number of layers 4 -> 35 output Wh
        self.encoder_net = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
        )

        self.to(self.device)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.encoder_net(obs)
        h = h.view(h.shape[0], -1)
        return h

class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, device, feature_dim=50, hidden_dim=1024):
        super(Actor, self).__init__()
        self.device = device
        self.log_std_min = -10
        self.log_std_max = 2

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                    nn.LayerNorm(feature_dim),
                                    nn.Tanh())

        self.net = nn.Sequential(
                                nn.Linear(feature_dim, hidden_dim),
                                nn.ReLU(),
                                nn.Linear(hidden_dim, hidden_dim),
                                nn.ReLU(),
                                nn.Linear(hidden_dim, 2*action_shape[0])
                                )

        self.apply(utils.weight_init)
        self.to(self.device)

    def forward(self, obs, evaluate=False, with_logprob=True):
        h = self.trunk(obs)

        mu, log_std = self.net(h).chunk(2, dim=-1)

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

class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, device, feature_dim=50, hidden_dim=1024):
        super(Critic, self).__init__()
        self.device = device

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                nn.LayerNorm(feature_dim),
                                nn.Tanh())

        self.Q1 = nn.Sequential(
                            nn.Linear(feature_dim + action_shape[0], hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, 1)
                            )
        self.Q2 = nn.Sequential(
                            nn.Linear(feature_dim + action_shape[0], hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, 1)
                            )

        self.apply(utils.weight_init)
        self.to(self.device)

    def forward(self, obs, action):
        h = self.trunk(obs)
        cat = T.cat([h, action], dim=-1)
        q1 = self.Q1(cat)
        q2 = self.Q2(cat)
        return q1, q2

# data augmentation Random shift
class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = T.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = T.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = T.randint(0,
                              2 * self.pad + 1,
                            size=(n, 1, 1, 2),
                            device=x.device,
                            dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                            grid,
                            padding_mode='zeros',
                            align_corners=False)

