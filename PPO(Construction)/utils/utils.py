import torch as T
import numpy as np
import random
from moviepy.editor import ImageSequenceClip
import os


def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

def ppo_iter(epoch, mini_batch_size, states, actions, values, log_probs, returns, advantages):
    batch_size = states.size(0)
    for _ in range(epoch):
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.choice(batch_size, mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids], values[rand_ids], log_probs[rand_ids], returns[rand_ids], advantages[rand_ids]


def random_seed(seed):
    if T.backends.cudnn.enabled:
        T.backends.cudnn.benchmark = False
        T.backends.cudnn.deterministic = True

    T.manual_seed(seed)
    T.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print('Using GPU : ', T.cuda.is_available() , ' |  Seed : ', seed)


def make_gif(policy, env, maxsteps=1000):
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
        action = policy.choose_action(state, 0)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        rewards.append(reward)
        t +=1
    print('Final reward :', np.sum(rewards))
    clip = ImageSequenceClip(steps, fps=30)
    if not os.path.isdir('gifs'):
        os.makedirs('gifs')
    clip.write_gif('gifs/{}.gif'.format(gif_name), fps=30)



def make_gif_for_train(policy, env, step_count, state_filter, maxsteps=1000):
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
        action = policy.get_action(state, state_filter=state_filter, deterministic=True)
        action = np.clip(action, env.action_space.low[0], env.action_space.high[0])
        action = action.reshape(len(action), )
        state, reward, done, _ = env.step(action)
        rewards.append(reward)
        t +=1
    print('Final reward :', np.sum(rewards))
    clip = ImageSequenceClip(steps, fps=30)
    if not os.path.isdir('gifs'):
        os.makedirs('gifs')
    clip.write_gif('gifs/{}.gif'.format(gif_name), fps=30)