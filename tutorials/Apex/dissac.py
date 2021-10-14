import numpy as np
import torch as T
import time
import ray
import gym
from gym.wrappers import RescaleAction

from actor_learner import Actor, Learner

import os
import multiprocessing
import copy
from ray.util import inspect_serializability

@ray.remote
class ReplayBuffer:
    def __init__(self, agent_args):
        self.agent_args = agent_args

        self.states = np.zeros([agent_args['buffer_size'], agent_args['n_states']], dtype=np.float32)
        self.next_states = np.zeros([agent_args['buffer_size'], agent_args['n_states']], dtype=np.float32)

        self.actions = np.zeros([agent_args['buffer_size'], agent_args['n_actions']], dtype=np.float32)

        self.rewards = np.zeros([agent_args['buffer_size']], dtype=np.float32)
        self.masks = np.zeros([agent_args['buffer_size']], dtype=np.float32)

        self.max_size = agent_args['buffer_size']
        self.ptr, self.cur_len, = 0, 0

    def store(self, state, action, reward, next_state, mask):

        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.masks[self.ptr] = mask

        self.ptr = (self.ptr + 1) % self.max_size
        self.cur_len = min(self.cur_len + 1, self.max_size)

    def sample_batch(self):

        index = np.random.choice(self.cur_len, self.agent_args['batch_size'], replace = False)

        return dict(
                    state = self.states[index],
                    action = self.actions[index],
                    reward = self.rewards[index],
                    next_state = self.next_states[index],
                    mask = self.masks[index],
                    )

    def __len__(self):

        return self.cur_len

    def ready(self):

        if self.cur_len >= self.agent_args['batch_size']:
            return True


@ray.remote
class ParameterServer(object):
    def __init__(self, weights, agent_args, weights_save_dir):
        self.agent_args = agent_args
        self.weights_save_dir = weights_save_dir
        if agent_args['restore']:
            self.weights = T.load(self.weights_save_dir)
        else:
            self.weights = weights.copy()

    def push(self, weights):
        self.weights = weights.copy()

    def pull(self):
        return self.weights

    def save_weights(self):
        T.save(self.weights, self.weights_save_dir)


@ray.remote(num_gpus=1, max_calls=1)
def worker_train(ps, replay_buffer, agent_args):

    agent = Learner(agent_args)

    weights = ray.get(ps.pull.remote())
    agent.set_weights(weights)

    while agent.total_step <= agent_args['time_steps']:

        agent.total_step += 1
        agent.learn(replay_buffer)

        if agent.total_step % 300 == 0:
            weights = agent.get_weights()
            ps.push.remote(weights)


@ray.remote
def worker_rollout(ps, replay_buffer, agent_args):

    env = gym.make(agent_args['env_name'])
    env = RescaleAction(env, -1, 1)

    agent = Actor(agent_args)

    weights = ray.get(ps.pull.remote())
    agent.set_weights(weights)

    max_ep_len = env.spec.max_episode_steps

    state = env.reset()
    done = False
    ep_len = 0

    for agent.total_step in range(1, agent_args['max_steps'] + 1):
        action = agent.select_exploration_action(state)

        next_state, reward, done, _ = env.step(action)
        ep_len += 1

        real_done = False if ep_len >= max_ep_len else done
        mask = 0.0 if real_done else agent_args['gamma']

        replay_buffer.store.remote(state, action, reward, next_state, mask)

        state = next_state

        if real_done or (ep_len >= max_ep_len):
            if replay_buffer.ready.remote():
                weights = ray.get(ps.pull.remote())
                agent.set_weights(weights)

            state = env.reset()
            done = False
            ep_len = 0


@ray.remote
def worker_test(ps, agent_args):

    env = gym.make(agent_args['env_name'])
    env = RescaleAction(env, -1, 1)

    best_score = env.reward_range[0]

    agent = Actor(agent_args)

    scores = []

    cnt = 1
    while cnt < agent_args['time_steps']:

        weights = ray.get(ps.pull.remote())
        agent.set_weights(weights)

        cnt += 1

        ave_ret = agent._evaluate_agent(env, agent, agent_args)
        scores.append(ave_ret)

        if cnt % 1000 == 0:
            print("test_reward:", ave_ret)

        if ave_ret > best_score:
            ps.save_weights.remote()
            print("****** weights saved! ******")
            best_score = ave_ret

        np.savetxt("./return.txt", scores, delimiter=",")
        time.sleep(5)


if __name__ == '__main__':
    from utils import Dict
    from configparser import ConfigParser
    from argparse import ArgumentParser

    parser = ArgumentParser('sac parameters')
    parser.add_argument("--algorithm", type=str, default = 'sac', help = 'algorithm to adjust (default : sac)')
    args = parser.parse_args()

    parser = ConfigParser()
    parser.read('config.ini')
    agent_args = Dict(parser, args.algorithm)

    env = gym.make(agent_args['env_name'])
    env = RescaleAction(env, -1, 1)

    agent_args['n_states'] = env.observation_space.shape[0]
    agent_args['n_actions'] = env.action_space.shape[0]
    agent_args['max_action'] = env.action_space.high[0]
    agent_args['low_action'] = env.action_space.low[0]
    agent_args['max_ep_len'] = env.spec.max_episode_steps

    weights_save_dir = os.path.join(agent_args['save_dir'] + '/' + agent_args['algorithm'] +'/' + agent_args['env_name'], 'sac_weights.pth')

    # Storage location creation
    if not os.path.exists(agent_args['save_dir']):
        os.mkdir(agent_args['save_dir'])

    model_path = agent_args['save_dir'] + '/' + agent_args['algorithm']
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    model_path = model_path + '/' + agent_args['env_name']
    if not os.path.exists(model_path):
        os.mkdir(model_path)


    ray.init()

    if agent_args['restore']:
        ps = ParameterServer.remote([], agent_args, weights_save_dir)
    else:
        net = Learner(agent_args)
        weights = net.get_weights()
        ps = ParameterServer.remote(weights, agent_args, weights_save_dir)

    replay_buffer = ReplayBuffer.remote(agent_args)

    # Start some training tasks.
    for _ in range(agent_args['num_workers']):
        worker_rollout.remote(ps, replay_buffer, agent_args)
        time.sleep(0.05)

    for _ in range(agent_args['num_learners']):
        worker_train.remote(ps, replay_buffer, agent_args)

    time.sleep(10)

    task_test = worker_test.remote(ps, agent_args)
    ray.wait([task_test,])
