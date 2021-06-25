import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CriticNetwork(nn.Module):
    def __init__(self, args):
        super(CriticNetwork, self).__init__()
        self.device = args.device

        self.max_action = args.high_action
        self.fc1 = nn.Linear(sum(args.obs_shape) + sum(args.action_shape), args.hidden_size_1)
        self.fc2 = nn.Linear(args.hidden_size_1, args.hidden_size_2)
        self.fc3 = nn.Linear(args.hidden_size_2, args.hidden_size_2)
        self.value = nn.Linear(args.hidden_size_2, 1)

        self.reset_parameters()

        self.optimizer = optim.Adam(self.parameters(), lr=args.critic_lr)
        self.to(self.device)

    def forward(self, state, action):
        state = T.cat(state, dim=1)
        for i in range(len(action)):
            action[i] /= self.max_action
        action = T.cat(action, dim=1)
        x = T.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.value(x)
        return q_value

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.fc3.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.value.weight, gain=nn.init.calculate_gain('relu'))