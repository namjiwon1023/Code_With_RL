import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy

class QNetwork(nn.Module):
    def __init__(self, n_states, n_actions, args):
        super(QNetwork, self).__init__()

        self.device = args.device
        self.checkpoint = os.path.join(args.save_dir+ '/' + args.env_name, 'D3QN.pth')

        self.feature = nn.Sequential(nn.Linear(n_states, args.hidden_size),
                                    nn.ReLU(),)

        self.advantage = nn.Sequential(
                                    nn.Linear(args.hidden_size, args.hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(args.hidden_size, n_actions),
                                        )


        self.value = nn.Sequential(
                                    nn.Linear(args.hidden_size, args.hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(args.hidden_size, 1),
                                        )

        self.reset_parameters(self.feature)
        self.reset_parameters(self.advantage)
        self.reset_parameters(self.value)

        self.optimizer = optim.Adam(self.parameters(), lr=args.critic_lr)

        self.to(self.device)

    def forward(self, state):
        feature = self.feature(state)
        advantage = self.advantage(feature)
        value = self.value(feature)

        out = value + advantage - advantage.mean(dim=-1, keepdim=True)

        return out

    def reset_parameters(self, Sequential, std=1.0, bias_const=1e-6):
        for layer in Sequential:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, std)
                nn.init.constant_(layer.bias, bias_const)

    def save_model(self):
        T.save(self.state_dict(), self.checkpoint)

    def load_model(self):
        self.load_state_dict(T.load(self.checkpoint))
