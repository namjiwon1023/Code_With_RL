import math

import torch as T
from torch import nn

import dmc2gym
import gym

gym.logger.set_level(40)


def make_dmc(domain_name, task_name, action_repeat, image_size=64):
    env = dmc2gym.make(
        domain_name=domain_name,
        task_name=task_name,
        visualize_reward=False,
        from_pixels=True,
        height=image_size,
        width=image_size,
        frame_skip=action_repeat,
    )
    setattr(env, 'action_repeat', action_repeat)
    return env


def create_input(features, actions):
    '''
        PI(a(t)|x(1:t),a(1:t-1))
        torch.cat([feature, action], dim=-1)
        torch.cat([next_feature, next_action], dim=-1)
        current state feature[0, t] | features[:, :-1].view(N, -1)
        next state feature[1, t + 1] | features[:, 1:].view(N, -1)
        action [0 , t] | actions[:, :-1].view(N, -1)
        next action [1, t + 1] | actions[:, 1:].view(N, -1)
    '''
    N = features.size(0)
    # Flatten sequence of features.
    feature = features[:, :-1].view(N, -1)
    next_feature = features[:, 1:].view(N, -1)
    # Flatten sequence of actions.
    action = actions[:, :-1].view(N, -1)
    next_action = actions[:, 1:].view(N, -1)
    # Concatenate feature and action.
    feature_actions = T.cat([feature, action], dim=-1)
    next_feature_actions = T.cat([next_feature, next_action], dim=-1)
    return feature_actions, next_feature_actions

def build_mlp(
    input_dim,
    output_dim,
    hidden_units=[128, 128],
    hidden_activation=nn.ReLU(),
    output_activation=None,
):
    layers = []
    units = input_dim
    for next_units in hidden_units:
        layers.append(nn.Linear(units, next_units))
        layers.append(hidden_activation)
        units = next_units
    layers.append(nn.Linear(units, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)

def calculate_kl_divergence(p_mean, p_std, q_mean, q_std):
    var_ratio = (p_std / q_std).pow_(2)
    t1 = ((p_mean - q_mean) / q_std).pow_(2)
    return 0.5 * (var_ratio + t1 - 1 - var_ratio.log())

def initialize_weight(m, std=1.0, bias_const=1e-6):
    '''
    In many papers, it is recommended to use the orthogonal method to initialize the network
    '''
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, std)
        nn.init.constant_(m.bias, bias_const)
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def calculate_gaussian_log_prob(log_std, noise):
    return (-0.5 * noise.pow(2) - log_std).sum(dim=-1, keepdim=True) - 0.5 * math.log(2 * math.pi) * log_std.size(-1)


def calculate_log_pi(log_std, noise, action):
    gaussian_log_prob = calculate_gaussian_log_prob(log_std, noise)
    return gaussian_log_prob - T.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)


def reparameterize(mean, log_std):
    noise = T.randn_like(mean)
    action = T.tanh(mean + noise * log_std.exp())
    return action, calculate_log_pi(log_std, noise, action)