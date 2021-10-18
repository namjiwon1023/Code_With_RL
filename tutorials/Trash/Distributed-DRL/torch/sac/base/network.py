import torch as T
import torch.nn as nn

class ActorSAC(nn.Module):
    def __init__(self, n_states, n_actions, args):
        super(ActorSAC, self).__init__()
        self.args = args
        self.min_log_std = args.min_log_std
        self.max_log_std = args.max_log_std

        self.feature = nn.Sequential(nn.Linear(n_states, args.hidden_units),
                                    nn.ReLU(),
                                    nn.Linear(args.hidden_units, args.hidden_units),
                                    nn.ReLU(),
                                    )

        self.log_std = nn.Linear(args.hidden_units, n_actions)
        self.mu = nn.Linear(args.hidden_units, n_actions)

        reset_parameters(self.feature)

        reset_single_layer_parameters(self.log_std)
        reset_single_layer_parameters(self.mu)

    def forward(self, state, evaluate=False, with_logprob=True):
        feature = self.feature(state)

        mu = self.mu(feature)
        log_std = self.log_std(feature)

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



class CriticTwin(nn.Module):
    def __init__(self, n_states, n_actions, args):
        super(CriticTwin, self).__init__()

        self.value1 = nn.Sequential(nn.Linear(n_states + n_actions, args.hidden_units),
                                    nn.ReLU(),
                                    nn.Linear(args.hidden_units, args.hidden_units),
                                    nn.ReLU(),
                                    nn.Linear(args.hidden_units, 1)
                                    )

        self.value2 = nn.Sequential(nn.Linear(n_states + n_actions, args.hidden_units),
                                    nn.ReLU(),
                                    nn.Linear(args.hidden_units, args.hidden_units),
                                    nn.ReLU(),
                                    nn.Linear(args.hidden_units, 1)
                                    )

        reset_parameters(self.value1)
        reset_parameters(self.value2)

    def forward(self, state, action):
        cat = T.cat((state, action), dim=-1)
        Q1 = self.value1(cat)
        Q2 = self.value2(cat)
        return Q1, Q2

def reset_parameters(Sequential, std=1.0, bias_const=1e-6):
    for layer in Sequential:
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, std)
            nn.init.constant_(layer.bias, bias_const)

def reset_single_layer_parameters(layer, std=1.0, bias_const=1e-6):
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)