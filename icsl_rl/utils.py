# Copyright (c) 2021: Zhiyuan Nan (namjw@hanyang.ac.kr).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import numpy as np
import random
from moviepy.editor import ImageSequenceClip
import os
import matplotlib.pyplot as plt
import yaml
import copy
import torch as T
import torch.nn as nn
import math

def _make_gif(policy, env, args, maxsteps=1000):
    envname = env.spec.id
    gif_name = '_'.join([envname])
    state = env.reset()
    done = False
    steps = []
    rewards = []
    t = 0
    while (not done) & (t< maxsteps):
        s = env.render('rgb_array')
        steps.append(s)
        if args.use_epsilon:
            action = policy.choose_action(state, 0)
        else:
            action = policy.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        rewards.append(reward)
        t +=1
    print('Final reward :', np.sum(rewards))
    clip = ImageSequenceClip(steps, fps=30)
    if not os.path.isdir('gifs'):
        os.makedirs('gifs')
    if not os.path.isdir('gifs' + '/' + args.algorithm):
        os.makedirs('gifs' + '/' + args.algorithm)
    clip.write_gif('gifs' + '/'  + args.algorithm + '/{}.gif'.format(gif_name), fps=30)

# total reward and average reward
def _plot(scores):
    z = [c+1 for c in range(len(scores))]
    running_avg = np.zeros(len(scores))
    for e in range(len(running_avg)):
        running_avg[e] = np.mean(scores[max(0, e-10):(e+1)])
    plt.cla()
    plt.title("Return")
    plt.grid(True)
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.plot(scores, "r-", linewidth=1.5, label="episode_reward")
    plt.plot(z, running_avg, "b-", linewidth=1.5, label="avg_reward")
    plt.legend(loc="best", shadow=True)
    plt.pause(0.1)
    plt.savefig('./sac.jpg')
    plt.show()

def _evaluate_agent(env, agent, args, n_starts=10):
    reward_sum = 0
    for _ in range(n_starts):
        done = False
        state = env.reset()
        while (not done):
            if args.evaluate:
                env.render()
            if args.use_epsilon:
                action = agent.choose_action(state, 0)
            else:
                action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            reward_sum += reward
            state = next_state
    return reward_sum / n_starts

def _store_expert_data(env, agent, args, n_starts=1000):
    episode_limit = env.spec.max_episode_steps
    transition = []
    for _ in range(n_starts):
        cur_episode_steps = 0
        done = False
        state = env.reset()
        while (not done):
            if args.evaluate:
                env.render()
            cur_episode_steps += 1
            if args.use_epsilon:
                action = agent.choose_action(state, 0)
            else:
                action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            real_done = False if cur_episode_steps >= episode_limit else done
            mask = 0.0 if real_done else args.gamma
            transition = [state, action, reward, next_state, mask]
            agent.memory.store_transition(transition)
            state = next_state
    return None

def _read_yaml(params):
    with open(params, encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    return config

class OUNoise:
    """Ornstein-Uhlenbeck process.
    Taken from Udacity deep-reinforcement-learning github repository:
    https://github.com/udacity/deep-reinforcement-learning/blob/master/
    ddpg-pendulum/ddpg_agent.py
    """

    def __init__(
        self,
        size: int,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
    ):
        """Initialize parameters and noise process."""
        self.state = np.float64(0.0)
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self) -> np.ndarray:
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array(
            [random.random() for _ in range(len(x))]
        )
        self.state = x + dx
        return self.state

def _target_soft_update(target, eval, tau):
    with T.no_grad():
        for t_p, l_p in zip(target.parameters(), eval.parameters()):
            t_p.data.copy_(tau * l_p.data + (1 - tau) * t_p.data)

def _target_net_update(target, eval):
    with T.no_grad():
        target.load_state_dict(eval.state_dict())

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

def calculate_gaussian_log_prob(log_std, noise):
    return (-0.5 * noise.pow(2) - log_std).sum(dim=-1, keepdim=True) - 0.5 * math.log(2 * math.pi) * log_std.size(-1)

def calculate_log_pi(log_std, noise, action):
    gaussian_log_prob = calculate_gaussian_log_prob(log_std, noise)
    return gaussian_log_prob - T.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

def reparameterize(mean, log_std):
    noise = T.randn_like(mean)
    action = T.tanh(mean + noise * log_std.exp())
    return action, calculate_log_pi(log_std, noise, action)

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

def conv2d_size_out(size, kernel_size, stride, padding):
    return ((size + 2 * padding - kernel_size) // stride) + 1