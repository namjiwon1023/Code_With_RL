import torch as T
import torch.nn as nn
import torch.optim as optim
import os
import random
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

class ActorNetwork(nn.Module):
    def __init__(self, n_states, n_actions, args, max_action=None):
        super(ActorNetwork, self).__init__()
        self.args = args
        self.device = args.device
        self.min_log_std = args.min_log_std
        self.max_log_std = args.max_log_std
        self.max_action = max_action
        self.checkpoint = os.path.join(args.save_dir + '/' + args.env_name, 'ppo_actor.pth')

        self.feature = nn.Sequential(nn.Linear(n_states, args.hidden_size),
                                    nn.ReLU()
                                    )
        self.mu = nn.Sequential(nn.Linear(args.hidden_size, n_actions),
                                nn.Tanh()
                                )
        self.log_std = nn.Sequential(nn.Linear(args.hidden_size, n_actions),
                                    nn.Tanh()
                                    )

        self.reset_parameters(self.feature)
        self.reset_parameters(self.mu)
        self.reset_parameters(self.log_std)

        self.optimizer = optim.Adam(self.parameters(), lr=args.actor_lr)

        self.to(self.device)

    def forward(self, state):
        feature = self.feature(state)
        mu = self.mu(feature)
        log_std = self.log_std(feature)
        log_std = self.min_log_std + 0.5 * (self.max_log_std - self.min_log_std) * (log_std + 1)
        std = T.exp(log_std)

        dist = Normal(mu, std)
        action = dist.sample()

        if self.max_action == None: return T.tanh(action), dist
        return self.max_action*T.tanh(action), dist


    def reset_parameters(self, Sequential, std=1.0, bias_const=1e-6):
        for layer in Sequential:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, std)
                nn.init.constant_(layer.bias, bias_const)


    def single_layer_reset_parameters(self, layer, std=1.0, bias_const=1e-6):
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, std)
            nn.init.constant_(layer.bias, bias_const)

    def save_model(self):
        T.save(self.state_dict(), self.checkpoint)

    def load_model(self):
        self.load_state_dict(T.load(self.checkpoint))
