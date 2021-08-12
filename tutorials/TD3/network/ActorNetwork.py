# Copyright (c) 2021: Zhiyuan Nan (namjw@hanyang.ac.kr).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class ActorNetwork(nn.Module):
    def __init__(self, n_states, n_actions, args, max_action=None):
        super(ActorNetwork, self).__init__()
        self.device = args.device
        self.max_action = max_action
        self.checkpoint = os.path.join(args.save_dir + '/' + args.env_name, 'TD3_actor.pth')

        self.actor = nn.Sequential(nn.Linear(n_states, args.hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(args.hidden_size, args.hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(args.hidden_size, n_actions),
                                    nn.Tanh())

        self.reset_parameters(self.actor)
        self.optimizer = optim.Adam(self.parameters(), lr=args.actor_lr)

        self.to(self.device)

    def forward(self, state):
        action = self.actor(state)
        if self.max_action == None: return action
        return self.max_action*action

    def reset_parameters(self, Sequential, std=1.0, bias_const=1e-6):
        for layer in Sequential:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, std)
                nn.init.constant_(layer.bias, bias_const)

    def save_model(self):
        T.save(self.state_dict(), self.checkpoint)

    def load_model(self):
        self.load_state_dict(T.load(self.checkpoint))
