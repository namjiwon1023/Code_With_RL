# Copyright (c) 2021: Zhiyuan Nan (namjw@hanyang.ac.kr).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import torch as T
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        self.args = args

        self.net = nn.Sequential(
                                nn.Linear(args['n_states'], args['ac_hidden_units']),
                                nn.ReLU(),
                                nn.Linear(args['ac_hidden_units'], args['ac_hidden_units']),
                                nn.ReLU(),
                                nn.Linear(args['ac_hidden_units'], args['n_actions']),
                                nn.Tanh(),
                                )

        self.apply(initialize_weight)
        self.to(args['device'])

    def forward(self, state):
        action = self.net(state)
        return action


class MultiCritic(nn.Module):
    def __init__(self, args):
        super(MultiCritic, self).__init__()
        self.nets = []

        for i in range(args['n_nets']):
            net = nn.Sequential(
                                nn.Linear(args['n_states'] + args['n_actions'], args['cri_hidden_units']),
                                nn.ReLU(),
                                nn.Linear(args['cri_hidden_units'], args['cri_hidden_units']),
                                nn.ReLU(),
                                nn.Linear(args['cri_hidden_units'], args['cri_hidden_units']),
                                nn.ReLU(),
                                nn.Linear(args['cri_hidden_units'], args['n_quantiles']),
                                )
            self.add_module(f'qf{i}', net)
            self.nets.append(net)

        self.apply(initialize_weight)
        self.to(args['device'])

    def forward(self, state, action):
        cat = T.cat([state, action], dim=1)
        quantiles = T.stack(tuple(net(cat) for net in self.nets), dim=1)
        return quantiles

def initialize_weight(m, std=1.0, bias_const=1e-6):
    '''
    linear layers initialization
    '''
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, std)
        nn.init.constant_(m.bias, bias_const)
    '''
    Convolution layers initialization
    '''
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)