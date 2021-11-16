# Copyright (c) 2021: Zhiyuan Nan (namjw@hanyang.ac.kr).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import torch as T
import torch.nn as nn
from torch.nn import functional as F

from utils import build_mlp, initialize_weight, reparameterize

class GaussianPolicy(nn.Module):
    """
    Policy parameterized as diagonal gaussian distribution.
    Actor Network
    input: feature * sequences(t) + sequences(t-1) * action
    PI(a(t+1)|x(1:t+1), a(1:t))
    """
    def __init__(self, action_shape, num_sequences, feature_dim, device, hidden_units=(256, 256), min_log_std=-20, max_log_std=2):
        super(GaussianPolicy, self).__init__()
        # NOTE: Conv layers are shared with the latent model.
        self.net = build_mlp(
            input_dim=num_sequences * feature_dim + (num_sequences - 1) * action_shape[0],
            output_dim=2 * action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=nn.ReLU(inplace=True),
            ).apply(initialize_weight)
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.device = device
        self.to(self.device)

    # Test functions
    def forward(self, feature_action):
        means = T.chunk(self.net(feature_action), 2, dim=-1)[0]
        return T.tanh(means)

    # Train functions
    def sample(self, feature_action):
        mean, log_std = T.chunk(self.net(feature_action), 2, dim=-1)
        action, log_pi = reparameterize(mean, log_std.clamp_(self.min_log_std, self.max_log_std))
        return action, log_pi

class QNetwork(nn.Module):
    """
    input : Z_dim = z1_dim + z2_dim = 256 + 32 = 288
    Q(z(t),a(t))
    """

    def __init__(
        self,
        action_shape,
        z1_dim,
        z2_dim,
        device,
        hidden_units=(256, 256),
    ):
        super(QNetwork, self).__init__()

        # Q(z(t),a(t)) ~ z(t) = z1 + z2
        # net1 , net2 ~ Double Q network ~ TwinnedQNetwork(SAC, TD3, DDQN, D3QN)
        self.net = build_mlp(
            input_dim=action_shape[0] + z1_dim + z2_dim,
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=nn.ReLU(inplace=True),
        ).apply(initialize_weight)

        self.device = device
        self.to(self.device)

    def forward(self, z, action):
        x = T.cat([z, action], dim=1)
        return self.net(x)
