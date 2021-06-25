import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CriticNetwork(nn.Module):
    def __init__(self, args):
        super(CriticNetwork, self).__init__()
        self.device = args.device

        self.max_action = args.high_action

        self.critic1 = nn.Sequential(nn.Linear(sum(args.obs_shape) + sum(args.action_shape), args.hidden_size_1),
                                    nn.ReLU(),
                                    nn.Linear(args.hidden_size_1, args.hidden_size_2),
                                    nn.ReLU(),
                                    nn.Linear(args.hidden_size_2, args.hidden_size_2),
                                    nn.ReLU(),
                                    nn.Linear(args.hidden_size_2, 1))


        self.critic2 = nn.Sequential(nn.Linear(sum(args.obs_shape) + sum(args.action_shape), args.hidden_size_1),
                                    nn.ReLU(),
                                    nn.Linear(args.hidden_size_1, args.hidden_size_2),
                                    nn.ReLU(),
                                    nn.Linear(args.hidden_size_2, args.hidden_size_2),
                                    nn.ReLU(),
                                    nn.Linear(args.hidden_size_2, 1))

        self.reset_parameters(self.critic1)
        self.reset_parameters(self.critic2)

        self.optimizer = optim.Adam(self.parameters(), lr=args.critic_lr)
        self.to(self.device)

    def forward(self, state, action):
        state = T.cat(state, dim=1)
        for i in range(len(action)):
            action[i] /= self.max_action
        action = T.cat(action, dim=1)

        x = T.cat([state, action], dim=1)
        Q1 = self.critic1(x)
        Q2 = self.critic2(x)

        return Q1, Q2

    def reset_parameters(self, Sequential, std=1.0, bias_const=1e-6):
        for layer in Sequential:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, std)
                nn.init.constant_(layer.bias, bias_const)