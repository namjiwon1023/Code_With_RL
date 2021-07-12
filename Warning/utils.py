import torch as T
import torch.nn as nn
import numpy as np
import random
from moviepy.editor import ImageSequenceClip
import os
from collections import deque
import matplotlib.pyplot as plt
import yaml

# target network hard update
def _target_net_update(eval_net, target_net):
    target_net.load_state_dict(eval_net.state_dict())

# target network soft update
def _target_soft_update(target_net, eval_net, args, tau=None):
    if tau == None:
        tau = args.tau
    with T.no_grad():
        for t_p, l_p in zip(target_net.parameters(), eval_net.parameters()):
            t_p.data.copy_(tau * l_p.data + (1 - tau) * t_p.data)

# generalized advantage estimator
def compute_gae(next_value, rewards, masks, values, gamma = 0.99, tau = 0.95,):
    values = values + [next_value]
    gae = 0
    returns = deque()

    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.appendleft(gae + values[step])

    return list(returns)

# Proximal Policy Optimization
def ppo_iter(epoch, mini_batch_size, states, actions, values, log_probs, returns, advantages,):
    batch_size = states.size(0)
    for _ in range(epoch):
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.choice(batch_size, mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids, :], values[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantages[rand_ids, :]

# Random Seed Settings
def _random_seed(seed):
    if T.backends.cudnn.enabled:
        T.backends.cudnn.benchmark = False
        T.backends.cudnn.deterministic = True

    T.manual_seed(seed)
    T.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print('Using GPU : ', T.cuda.is_available() , ' |  Seed : ', seed)


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


def _save_model(net, dirpath):
    print('------ Save model ------')
    T.save(net.state_dict(), dirpath)


def _load_model(net, dirpath):
    print('------ load model ------')
    net.load_state_dict(T.load(dirpath))

def _read_yaml(params):
    with open(params, encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    return config

# network layers parameters resetting
def reset_parameters(Sequential, std=1.0, bias_const=1e-6):
    for layer in Sequential:
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, std)
            nn.init.constant_(layer.bias, bias_const)

def reset_single_layer_parameters(layer, std=1.0, bias_const=1e-6):
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, std)
            nn.init.constant_(layer.bias, bias_const)