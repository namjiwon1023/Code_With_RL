import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -5
epsilon = 1e-6

device = T.device('cuda' if T.cuda.is_available() else 'cpu')

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        T.nn.init.xavier_uniform_(m.weight, gain=1)
        T.nn.init.constant_(m.bias, 0)

'''SAC with D2RL Network'''
class D2RLQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, num_layers):
        super(D2RLQNetwork, self).__init__()

        in_dim = num_inputs+num_actions+hidden_dim
        # Q1 architecture
        self.l1_1 = nn.Linear(num_inputs+num_actions, hidden_dim)
        self.l1_2 = nn.Linear(in_dim, hidden_dim)

        self.l2_1 = nn.Linear(num_inputs+num_actions, hidden_dim)
        self.l2_2 = nn.Linear(in_dim, hidden_dim)

        if num_layers > 2:
            self.l1_3 = nn.Linear(in_dim, hidden_dim)
            self.l1_4 = nn.Linear(in_dim, hidden_dim)

            self.l2_3 = nn.Linear(in_dim, hidden_dim)
            self.l2_4 = nn.Linear(in_dim, hidden_dim)

        if num_layers > 4:
            self.l1_5 = nn.Linear(in_dim, hidden_dim)
            self.l1_6 = nn.Linear(in_dim, hidden_dim)

            self.l2_5 = nn.Linear(in_dim, hidden_dim)
            self.l2_6 = nn.Linear(in_dim, hidden_dim)

        if num_layers == 8:
            self.l1_7 = nn.Linear(in_dim, hidden_dim)
            self.l1_8 = nn.Linear(in_dim, hidden_dim)

            self.l2_7 = nn.Linear(in_dim, hidden_dim)
            self.l2_8 = nn.Linear(in_dim, hidden_dim)

        self.out1 = nn.Linear(hidden_dim, 1)
        self.out2 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

        self.num_layers = num_layers

    def forward(self, state, action):
        xu = T.cat([state, action], 1)

        x1 = F.relu(self.l1_1(xu))
        x2 = F.relu(self.l2_1(xu))
        x1 = T.cat([x1, xu], dim=1)
        x2 = T.cat([x2, xu], dim=1)

        x1 = F.relu(self.l1_2(x1))
        x2 = F.relu(self.l2_2(x2))
        if not self.num_layers == 2:
            x1 = T.cat([x1, xu], dim=1)
            x2 = T.cat([x2, xu], dim=1)

        if self.num_layers > 2:
            x1 = F.relu(self.l1_3(x1))
            x2 = F.relu(self.l2_3(x2))
            x1 = T.cat([x1, xu], dim=1)
            x2 = T.cat([x2, xu], dim=1)

            x1 = F.relu(self.l1_4(x1))
            x2 = F.relu(self.l2_4(x2))
            if not self.num_layers == 4:
                x1 = T.cat([x1, xu], dim=1)
                x2 = T.cat([x2, xu], dim=1)

        if self.num_layers > 4:
            x1 = F.relu(self.l1_5(x1))
            x2 = F.relu(self.l2_5(x2))
            x1 = T.cat([x1, xu], dim=1)
            x2 = T.cat([x2, xu], dim=1)

            x1 = F.relu(self.l1_6(x1))
            x2 = F.relu(self.l2_6(x2))
            if not self.num_layers == 6:
                x1 = T.cat([x1, xu], dim=1)
                x2 = T.cat([x2, xu], dim=1)

        if self.num_layers == 8:
            x1 = F.relu(self.l1_7(x1))
            x2 = F.relu(self.l2_7(x2))
            x1 = T.cat([x1, xu], dim=1)
            x2 = T.cat([x2, xu], dim=1)

            x1 = F.relu(self.l1_8(x1))
            x2 = F.relu(self.l2_8(x2))

        x1 = self.out1(x1)
        x2 = self.out2(x2)

        return x1, x2


class D2RLGaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, num_layers, action_space=None):
        super(D2RLGaussianPolicy, self).__init__()
        self.num_layers = num_layers

        in_dim = hidden_dim+num_inputs
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(in_dim, hidden_dim)


        if num_layers > 2:
            self.linear3 = nn.Linear(in_dim, hidden_dim)
            self.linear4 = nn.Linear(in_dim, hidden_dim)
        if num_layers > 4:
            self.linear5 = nn.Linear(in_dim, hidden_dim)
            self.linear6 = nn.Linear(in_dim, hidden_dim)
        if num_layers == 8:
            self.linear7 = nn.Linear(in_dim, hidden_dim)
            self.linear8 = nn.Linear(in_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = T.tensor(1.)
            self.action_bias = T.tensor(0.)
        else:
            self.action_scale = T.FloatTensor(
                (action_space.high - action_space.low) / 2.).to(device)
            self.action_bias = T.FloatTensor(
                (action_space.high + action_space.low) / 2.).to(device)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = T.cat([x, state], dim=1)

        x = F.relu(self.linear2(x))

        if self.num_layers > 2:
            x = T.cat([x, state], dim=1)
            x = F.relu(self.linear3(x))

            x = T.cat([x, state], dim=1)
            x = F.relu(self.linear4(x))

        if self.num_layers > 4:
            x = T.cat([x, state], dim=1)
            x = F.relu(self.linear5(x))

            x = T.cat([x, state], dim=1)
            x = F.relu(self.linear6(x))

        if self.num_layers == 8:
            x = T.cat([x, state], dim=1)
            x = F.relu(self.linear7(x))

            x = T.cat([x, state], dim=1)
            x = F.relu(self.linear8(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = T.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = T.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= T.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = T.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

'''DDPG TD3 with D2RL Network'''

class D2RLActor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(D2RLActor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256+state_dim, 256)
        self.l3 = nn.Linear(256+state_dim, 256)
        self.l4 = nn.Linear(256+state_dim, 256)
        self.l5 = nn.Linear(256, action_dim)

        self.max_action = max_action


    def forward(self, state):
        a = F.relu(self.l1(state))

        a = T.cat([a, state], 1)
        a = F.relu(self.l2(a))

        a = T.cat([a, state], 1)
        a = F.relu(self.l3(a))

        a = T.cat([a, state], 1)
        a = F.relu(self.l4(a))

        return self.max_action * T.tanh(self.l5(a))

class D2RLCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(D2RLCritic, self).__init__()

        in_dim = state_dim + action_dim

        # Q1 architecture
        self.l1 = nn.Linear(in_dim, 256)
        self.l2 = nn.Linear(256+in_dim, 256)
        self.l3 = nn.Linear(256+in_dim, 256)
        self.l4 = nn.Linear(256+in_dim, 256)
        self.l5 = nn.Linear(256, 1)

        # Q2 architecture
        self.l6 = nn.Linear(in_dim, 256)
        self.l7 = nn.Linear(256+in_dim, 256)
        self.l8 = nn.Linear(256+in_dim, 256)
        self.l9 = nn.Linear(256+in_dim, 256)
        self.l0 = nn.Linear(256, 1)


    def forward(self, state, action):
        sa = T.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))

        q1 = T.cat([q1, sa], 1)
        q1 = F.relu(self.l2(q1))

        q1 = T.cat([q1, sa], 1)
        q1 = F.relu(self.l3(q1))

        q1 = T.cat([q1, sa], 1)
        q1 = F.relu(self.l4(q1))

        q1 = self.l5(q1)

        q2 = F.relu(self.l6(sa))

        q2 = T.cat([q2, sa], 1)
        q2 = F.relu(self.l7(q2))

        q2 = T.cat([q2, sa], 1)
        q2 = F.relu(self.l8(q2))

        q2 = T.cat([q2, sa], 1)
        q2 = F.relu(self.l9(q2))

        q2 = self.l0(q2)
        return q1, q2


    def Q1(self, state, action):
        sa = T.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))

        q1 = T.cat([q1, sa], 1)
        q1 = F.relu(self.l2(q1))

        q1 = T.cat([q1, sa], 1)
        q1 = F.relu(self.l3(q1))

        q1 = T.cat([q1, sa], 1)
        q1 = F.relu(self.l4(q1))

        q1 = self.l5(q1)
        return q1

