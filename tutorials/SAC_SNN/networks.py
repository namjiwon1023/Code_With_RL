# Copyright (c) 2021: Zhiyuan Nan (namjw@hanyang.ac.kr).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import torch as T
import torch.nn as nn
from torch.distributions import Normal

from spikingjelly.clock_driven import neuron

# Neural Network
class NonSpikingLIFNode(neuron.LIFNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, dv: T.Tensor):
        self.neuronal_charge(dv)
        # self.neuronal_fire()
        # self.neuronal_reset()
        return self.v

class SpikingActorSAC(nn.Module):
    def __init__(self, args):
        super(SpikingActorSAC, self).__init__()
        self.args = args

        self.min_log_std = args['min_log_std']
        self.max_log_std = args['max_log_std']

        self.Spiking_T = args['spiking_t']

        self.net = nn.Sequential(
                                nn.Linear(args['n_states'], args['hidden_units']),
                                neuron.IFNode(),
                                nn.Linear(args['hidden_units'], args['hidden_units']),
                                neuron.IFNode(),
                                nn.Linear(args['hidden_units'], 2*args['n_actions']),
                                NonSpikingLIFNode(tau=2.0),
                                )

        self.apply(initialize_weight)
        self.to(args['device'])

    def forward(self, state, evaluate=False, with_logprob=True):
        for t in range(self.Spiking_T):
            self.net(state)
        mu, log_std = self.net[-1].v.chunk(2, dim=-1)

        log_std = T.clamp(log_std, self.min_log_std, self.max_log_std)
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


class SpikingCriticTwin(nn.Module):
    def __init__(self, args):
        super(SpikingCriticTwin, self).__init__()
        self.Spiking_T = args['spiking_t']

        self.value1 = nn.Sequential(
                                    nn.Linear(args['n_states'] + args['n_actions'], args['hidden_units']),
                                    neuron.IFNode(),
                                    nn.Linear(args['hidden_units'], args['hidden_units']),
                                    neuron.IFNode(),
                                    nn.Linear(args['hidden_units'], 1),
                                    NonSpikingLIFNode(tau=2.0)
                                    )

        self.value2 = nn.Sequential(
                                    nn.Linear(args['n_states'] + args['n_actions'], args['hidden_units']),
                                    neuron.IFNode(),
                                    nn.Linear(args['hidden_units'], args['hidden_units']),
                                    neuron.IFNode(),
                                    nn.Linear(args['hidden_units'], 1),
                                    NonSpikingLIFNode(tau=2.0)
                                    )

        self.apply(initialize_weight)

        self.to(args['device'])

    def forward(self, state, action):
        cat = T.cat([state, action], dim=1)

        for t in range(self.Spiking_T):
            self.value1(cat)
            self.value2(cat)

        Q1 = self.value1[-1].v
        Q2 = self.value2[-1].v

        return Q1, Q2

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