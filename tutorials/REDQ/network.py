import torch as T
import torch.nn as nn
import os
import numpy as np
from torch.distributions import Normal, Distribution

class Actor(nn.Module):
    def __init__(self, n_states, n_actions, args):
        super(Actor, self).__init__()
        self.args = args
        self.device = args.device
        self.log_std_min = args.log_std_min
        self.log_std_max = args.log_std_max

        self.net = nn.Sequential(
                                nn.Linear(n_states, args.ac_hidden_size),
                                nn.ReLU(),
                                nn.Linear(args.ac_hidden_size, args.ac_hidden_size),
                                nn.ReLU(),
                                nn.Linear(args.ac_hidden_size, 2*n_actions)
                                )
        self.to(self.device)

    def forward(self, state, evaluate=False, with_logprob=True):

        mu, log_std = self.net(state).chunk(2, dim=-1)

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
        self.Q = nn.Sequential(nn.Linear(n_states + n_actions, args.cri_hidden_size),
                                nn.ReLU(),
                                nn.Linear(args.cri_hidden_size, args.cri_hidden_size),
                                nn.ReLU(),
                                nn.Linear(args.cri_hidden_size, 1))
        self.to(self.device)

    def forward(self, state, action):
        cat = T.cat((state, action), dim=-1)
        return self.Q(cat)

class TanhNormal(Distribution):
    """
    Represent distribution of X where
        X ~ tanh(Z)
        Z ~ N(mean, std)
    Note: this is not very numerically stable.
    """
    def __init__(self, normal_mean, normal_std, epsilon=1e-6):
        """
        :param normal_mean: Mean of the normal distribution
        :param normal_std: Std of the normal distribution
        :param epsilon: Numerical stability epsilon when computing log-prob.
        """
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon

    def log_prob(self, value, pre_tanh_value=None):
        """
        return the log probability of a value
        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        # use arctanh formula to compute arctanh(value)
        if pre_tanh_value is None:
            pre_tanh_value = torch.log(
                (1+value) / (1-value)
            ) / 2
        return self.normal.log_prob(pre_tanh_value) - \
               torch.log(1 - value * value + self.epsilon)

    def sample(self, return_pretanh_value=False):
        """
        Gradients will and should *not* pass through this operation.
        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        z = self.normal.sample().detach()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def rsample(self, return_pretanh_value=False):
        """
        Sampling in the reparameterization case.
        Implement: tanh(mu + sigma * eksee)
        with eksee~N(0,1)
        z here is mu+sigma+eksee
        """
        z = (
            self.normal_mean +
            self.normal_std *
            Normal( ## this part is eksee~N(0,1)
                torch.zeros(self.normal_mean.size()),
                torch.ones(self.normal_std.size())
            ).sample()
        )
        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)