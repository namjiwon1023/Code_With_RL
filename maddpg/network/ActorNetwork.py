import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ActorNetwork(nn.Module):
    def __init__(self, args, agent_id):
        super(ActorNetwork, self).__init__()
        self.device = args.device

        self.max_action = args.high_action
        self.fc1 = nn.Linear(args.obs_shape[agent_id], args.hidden_size_1)
        self.fc2 = nn.Linear(args.hidden_size_1, args.hidden_size_2)
        self.fc3 = nn.Linear(args.hidden_size_2, args.hidden_size_2)
        self.pi = nn.Linear(args.hidden_size_2, args.action_shape[agent_id])

        self.reset_parameters()

        self.optimizer = optim.Adam(self.parameters(), lr=args.actor_lr)
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * T.tanh(self.pi(x))

        return actions

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.fc3.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.pi.weight, gain=nn.init.calculate_gain('relu'))