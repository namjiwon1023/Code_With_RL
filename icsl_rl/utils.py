import numpy as np
import random
from moviepy.editor import ImageSequenceClip
import os
import matplotlib.pyplot as plt
import yaml
import copy
import torch as T
import torch.nn as nn

# from https://github.com/openai/spinningup
######################################
def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

def combined_shape(length, shape=None):    # np.zeros(combined_shape(size, act_dim), dtype=np.float32)
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

#############################################

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

def _make_gif_for_train(policy, env, args, step_count, maxsteps=1000):
    envname = env.spec.id
    gif_name = '_'.join([envname, str(step_count)])
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
        state, reward, done, _ = env.step(action)
        rewards.append(reward)
        t +=1
    print('Final reward :', np.sum(rewards))
    clip = ImageSequenceClip(steps, fps=30)
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
