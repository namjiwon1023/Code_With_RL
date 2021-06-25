import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

class ActorNetwork(nn.Module):
    def __init__(self, args, agent_id):
        super(ActorNetwork, self).__init__()
        self.args = args
        self.device = args.device

        self.max_action = args.high_action

        self.feature = nn.Sequential(nn.Linear(args.obs_shape[agent_id], args.hidden_size_1),
                                    nn.ReLU(),
                                    nn.Linear(args.hidden_size_1, args.hidden_size_2),
                                    nn.ReLU(),
                                    nn.Linear(args.hidden_size_2, args.hidden_size_2),
                                    nn.ReLU())

        self.log_std = nn.Linear(args.hidden_size_2, args.action_shape[agent_id])
        self.mu = nn.Linear(args.hidden_size_2, args.action_shape[agent_id])

        self.reset_parameters(self.feature)
        self.single_init(self.log_std)
        self.single_init(self.mu)

        self.optimizer = optim.Adam(self.parameters(), lr=args.actor_lr)
        self.to(self.device)

    def forward(self, x):

        feature = self.feature(x)

        mu = self.mu(feature)
        log_std = self.log_std(feature)
        log_std = T.clamp(log_std, self.args.min_log_std, self.args.max_log_std)
        std = T.exp(log_std)

        dist = Normal(mu, std)
        z = dist.rsample()

        if self.args.evaluate:
            action = mu.tanh()
        else:
            action = z.tanh()

        if self.args.with_logprob:
            log_prob = dist.log_prob(z) - T.log(1 - action.pow(2) + 1e-7)
            log_prob = log_prob.sum(-1, keepdim=True)
        else:
            log_prob = None

        actions = self.max_action * action

        return actions, log_prob

    def reset_parameters(self, Sequential, std=1.0, bias_const=1e-6):
        for layer in Sequential:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, std)
                nn.init.constant_(layer.bias, bias_const)

    def single_init(self, layer, std=1.0, bias_const=1e-6):
        if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, std)
                nn.init.constant_(layer.bias, bias_const)