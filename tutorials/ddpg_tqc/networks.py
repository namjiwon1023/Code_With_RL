# Copyright (c) 2021: Zhiyuan Nan (namjw@hanyang.ac.kr).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import torch as T
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        self.args = args
        self.in_dim = args['n_states'] + args['ac_hidden_units']

        self.fc1 = nn.Linear(args['n_states'], args['ac_hidden_units'])
        self.fc2 = nn.Linear(self.in_dim, args['ac_hidden_units'])
        self.fc3 = nn.Linear(self.in_dim, args['ac_hidden_units'])
        self.fc4 = nn.Linear(args['ac_hidden_units'], args['n_actions'])

        self.apply(initialize_weight)
        self.to(args['device'])

    def forward(self, state):
        x = F.relu(self.fc1(state))

        x = T.cat([x, state], dim=1)
        x = F.relu(self.fc2(x))

        x = T.cat([x, state], dim=1)
        x = F.relu(self.fc3(x))

        action = T.tanh(self.fc4(x))

        return action


class MultiCritic(nn.Module):
    def __init__(self, args):
        super(MultiCritic, self).__init__()
        self.nets = []
        self.in_dim = args['n_states'] + args['n_actions'] + args['cri_hidden_units']

        for i in range(args['n_nets']):
            net = nn.Sequential(
                                nn.Linear(args['n_states'] + args['n_actions'], args['cri_hidden_units']),
                                nn.Linear(self.in_dim, args['cri_hidden_units']),
                                nn.Linear(self.in_dim, args['cri_hidden_units']),
                                nn.Linear(args['cri_hidden_units'], args['n_quantiles']),
                                )
            self.add_module(f'qf{i}', net)
            self.nets.append(net)

        self.apply(initialize_weight)
        self.to(args['device'])

    def forward(self, state, action):
        cat = T.cat([state, action], dim=1)
        quantile = list()
        for net in self.nets:
            x = F.relu(net[0](cat))
            x = T.cat([x, cat],dim=1)
            x = F.relu(net[1](x))
            x = T.cat([x, cat],dim=1)
            x = F.relu(net[2](x))
            out = net[3](x)
            quantile.append(out)
        quantiles = T.stack(quantile, dim=1)
        # print('quantiles : {} | shape : {} '.format(quantiles, quantiles.shape))
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
