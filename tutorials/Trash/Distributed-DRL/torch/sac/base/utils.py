import numpy as np
import random
import torch as T
import torch.nn as nn
import math

def _evaluate_agent(env, agent, args, n_starts=10, render=False):
    reward_sum = 0
    for _ in range(n_starts):
        done = False
        state = env.reset()
        while (not done):
            if render:
                env.render()
            with eval_mode(agent):
                action = agent.choose_action(state, evaluate=True)
            next_state, reward, done, _ = env.step(action)
            reward_sum += reward
            state = next_state
    return reward_sum / n_starts

def _target_soft_update(target, eval, tau):
    for t_p, l_p in zip(target.parameters(), eval.parameters()):
        t_p.data.copy_(tau * l_p.data + (1 - tau) * t_p.data)

def _grad_false(network):
    for param in network.parameters():
        param.requires_grad = False


# model save functions
def _save_model(net, dirpath):
    T.save(net.state_dict(), dirpath)

# model load functions
def _load_model(net, dirpath):
    net.load_state_dict(T.load(dirpath))

# Random Seed Settings
def set_seed_everywhere(seed):
    T.manual_seed(seed)
    if T.cuda.is_available():
        T.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def reset_parameters(Sequential, std=1.0, bias_const=1e-6):
    for layer in Sequential:
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, std)
            nn.init.constant_(layer.bias, bias_const)

def reset_single_layer_parameters(layer, std=1.0, bias_const=1e-6):
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)

class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False