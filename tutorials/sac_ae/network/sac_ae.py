import torch as T
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import math
from torch.distributions import Normal
from network.encoder import Encoder
from utils.utils import weight_init

LOG_FREQ = 10000
def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = T.tanh(mu)
    if pi is not None:
        pi = T.tanh(pi)
    if log_pi is not None:
        log_pi -= T.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi

class Actor(nn.Module):
    def __init__(self, obs_shape, action_shape, feature_dim, device, log_std_min=-10, log_std_max=2):
        super(Actor, self).__init__()
        self.encoder = Encoder(obs_shape, feature_dim, device)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.net = nn.Sequential(
                    nn.Linear(feature_dim, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 2 * action_shape[0])
                    )

        self.outputs = dict()
        self.apply(weight_init)
        self.to(device)

    def forward(
        self, obs, compute_pi=True, compute_log_pi=True, detach_encoder=False
    ):
        obs = self.encoder(obs, detach=detach_encoder)

        mu, log_std = self.net(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = T.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        self.outputs['mu'] = mu
        self.outputs['std'] = log_std.exp()

        if compute_pi:
            std = log_std.exp()
            noise = T.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std

    # def forward(self, obs, evaluate=False, with_logprob=True, detach_encoder=False):
    #     obs = self.encoder(obs, detach=detach_encoder)

    #     mu, log_std = self.net(obs).chunk(2, dim=-1)

    #     log_std = T.clamp(log_std, self.log_std_min, self.log_std_max)
    #     std = T.exp(log_std)

    #     self.outputs['mu'] = mu
    #     self.outputs['std'] = std

    #     dist = Normal(mu, std)
    #     z = dist.rsample()

    #     if evaluate:
    #         action = mu.tanh()
    #     else:
    #         action = z.tanh()

    #     if with_logprob:
    #         log_prob = dist.log_prob(z) - T.log(1 - action.pow(2) + 1e-7)
    #         log_prob = log_prob.sum(-1, keepdim=True)
    #     else:
    #         log_prob = None

    #     return action, log_prob

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_actor/%s_hist' % k, v, step)

        L.log_param('train_actor/fc1', self.net[0], step)
        L.log_param('train_actor/fc2', self.net[2], step)
        L.log_param('train_actor/fc3', self.net[4], step)


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

        self.outputs = dict()
        self.apply(weight_init)
        self.to(device)

    def forward(self, obs, action, detach_encoder=False):
        obs = self.encoder(obs, detach=detach_encoder)

        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        self.encoder.log(L, step, log_freq)

        for k, v in self.outputs.items():
            L.log_histogram('train_critic/%s_hist' % k, v, step)

        for i in range(3):
            L.log_param('train_critic/q1_fc%d' % i, self.Q1.net[i * 2], step)
            L.log_param('train_critic/q2_fc%d' % i, self.Q2.net[i * 2], step)
