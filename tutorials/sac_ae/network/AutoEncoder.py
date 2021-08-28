# Copyright (c) 2021: Zhiyuan Nan (namjw@hanyang.ac.kr).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import torch as T
import torch.nn as nn
from utils.utils import tie_weights

'''
OUT_DIM :
number of layers 2 -> 39
number of layers 4 -> 35
number of layers 6 -> 31
'''

class Encoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, device):
        super(Encoder, self).__init__()
        assert len(obs_shape) == 3   # C， W， H -> len(3, 84, 84) = 3

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

        # num_filters * out_dim * out_dim -> feature_dim
        self.fc = nn.Linear(32 * 35 * 35, feature_dim)
        self.ln = nn.LayerNorm(feature_dim)

        self.to(device)

    def get_hidden(self, obs):
        obs = obs / 255.
        x = self.encoder_net(obs)
        h = x.view(x.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.get_hidden(obs)
        if detach:
            h = h.detach()
        h_fc = self.fc(h)
        h_norm = self.ln(h_fc)
        out = T.tanh(h_norm)
        return out

    def sharing_parameters_actor_critic_encoder(self, source):
        # actor encoder and critic encoder sharing parameters.
        for i in range(int(len(self.encoder_net)/2)):
            tie_weights(src=source.encoder_net[i*2], trg=self.encoder_net[i*2])

class Decoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, device):
        super(Decoder, self).__init__()

        # feature_dim -> num_filters * out_dim * out_dim
        self.fc = nn.Linear(feature_dim, 32 * 35 * 35)

        # num_filters * out_dim * out_dim -> 3, 84, 84
        self.decoder_net = nn.Sequential(
                        nn.ConvTranspose2d(32, 32, 3, stride=1),
                        nn.ReLU(),
                        nn.ConvTranspose2d(32, 32, 3, stride=1),
                        nn.ReLU(),
                        nn.ConvTranspose2d(32, 32, 3, stride=1),
                        nn.ReLU(),
                        nn.ConvTranspose2d(32, obs_shape[0], 3, stride=2, output_padding=1),
                        )

        self.to(device)

    def forward(self, h):
        h = T.relu(self.fc(h))
        x = h.view(-1, 32, 35, 35)

        obs = self.decoder_net(x)

        return obs