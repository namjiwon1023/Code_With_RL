import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import copy
import utils
from torch.distributions import Normal

class Encoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, device):
        super().__init__()
        assert len(obs_shape) == 3   # C， W， H -> len(3, 84, 84) = 3

        self.output_logits = False
        self.feature_dim = feature_dim
        self.device = device

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
        # # num_filters * out_dim * out_dim -> feature_dim
        self.head = nn.Sequential(
            nn.Linear(32 * 35 * 35, self.feature_dim),
            nn.LayerNorm(self.feature_dim))

        self.to(self.device)

    def get_hidden(self, obs):
        obs = obs / 255.
        x = self.encoder_net(obs)
        h = x.view(x.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.get_hidden(obs)
        if detach:
            h = h.detach()
        out = self.head(h)
        if not self.output_logits:
            out = T.tanh(out)
        return out

    def sharing_parameters_actor_critic_encoder(self, source):
        # actor encoder and critic encoder sharing parameters.
        for i in range(int(len(self.encoder_net)/2)):
            utils.tie_weights(src=source.encoder_net[i*2], trg=self.encoder_net[i*2])

class Actor(nn.Module):
    def __init__(self, obs_shape, action_shape, device, feature_dim=50, hidden_dim=1024):
        super(Actor, self).__init__()
        self.device = device
        self.encoder = Encoder(obs_shape, feature_dim, device)
        self.log_std_min = -10
        self.log_std_max = 2

        self.net = nn.Sequential(
                                nn.Linear(self.encoder.feature_dim, hidden_dim),
                                nn.ReLU(),
                                nn.Linear(hidden_dim, hidden_dim),
                                nn.ReLU(),
                                nn.Linear(hidden_dim, 2*action_shape[0])
                                )

        self.apply(utils.weight_init)
        self.to(self.device)

    def forward(self, obs, detach_encoder=False, evaluate=False, with_logprob=True):
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

        return action, log_prob

class Critic(nn.Module):
    def __init__(self, obs_shape, action_shape, device, feature_dim=50, hidden_dim=1024):
        super(Critic, self).__init__()
        self.device = device

        self.encoder = Encoder(obs_shape, feature_dim, device)

        self.Q1 = nn.Sequential(
                            nn.Linear(self.encoder.feature_dim + action_shape[0], hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, 1)
                            )
        self.Q2 = nn.Sequential(
                            nn.Linear(self.encoder.feature_dim + action_shape[0], hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, 1)
                            )

        self.apply(utils.weight_init)
        self.to(self.device)

    def forward(self, obs, action, detach_encoder=False):
        assert obs.size(0) == action.size(0)
        obs = self.encoder(obs, detach=detach_encoder)
        cat = T.cat([obs, action], dim=-1)
        q1 = self.Q1(cat)
        q2 = self.Q2(cat)
        return q1, q2


