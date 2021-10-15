import numpy as np
import torch as T
import time
import ray
import gym
from gym.wrappers import RescaleAction
from actor_learner import Actor, Learner
import random
import os
import copy
from torch.utils.tensorboard import SummaryWriter

@ray.remote
class ReplayBuffer:
    def __init__(self, args):
        self.args = args
        self.states = np.zeros([args['buffer_size'], args['n_states']], dtype=np.float32)
        self.next_states = np.zeros([args['buffer_size'], args['n_states']], dtype=np.float32)
        self.actions = np.zeros([args['buffer_size'], args['n_actions']], dtype=np.float32)
        self.rewards = np.zeros([args['buffer_size']], dtype=np.float32)
        self.masks = np.zeros([args['buffer_size']], dtype=np.float32)
        self.max_size = args['buffer_size']
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
        index = np.random.choice(self.cur_len, self.args['batch_size'], replace = False)
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
        if self.cur_len >= self.args['batch_size']:
            return True

@ray.remote
class ParameterServer(object):
    def __init__(self, weights, args, weights_save_dir):
        self.args = args
        self.weights_save_dir = weights_save_dir
        if args['restore']:
            self.weights = T.load(self.weights_save_dir)
        else:
            self.weights = copy.deepcopy(weights)

    def push(self, weights):
        self.weights = copy.deepcopy(weights)

    def pull(self):
        return copy.deepcopy(self.weights)

    def save_weights(self):
        T.save(self.weights, self.weights_save_dir)


@ray.remote(num_gpus=1, max_calls=1)
def worker_train(ps, replay_buffer, args):

    writer = SummaryWriter('./logs/' + args['algorithm'])

    T.manual_seed(args['seed'])
    T.cuda.manual_seed(args['seed'])
    T.cuda.manual_seed_all(args['seed'])
    np.random.seed(args['seed'])
    random.seed(args['seed'])

    agent = Learner(args)

    weights = ray.get(ps.pull.remote())
    agent.set_weights(weights)

    cnt = 1

    while True:

        agent.learn(replay_buffer, writer)

        if cnt % 300 == 0:
            print('Weights push to PS !!!')
            weights = agent.get_weights()
            ps.push.remote(weights)

        cnt += 1

@ray.remote
def worker_rollout(ps, replay_buffer, args, worker_id):

    env = gym.make(args['env_name'])
    env = RescaleAction(env, -1, 1)

    T.manual_seed(args['seed'] + worker_id * 1000)
    np.random.seed(args['seed'] + worker_id * 1000)
    random.seed(args['seed'] + worker_id * 1000)

    env.seed(args['seed'] + worker_id * 1000)
    env.action_space.np_random.seed(args['seed'] + worker_id * 1000)

    agent = Actor(args)

    weights = ray.get(ps.pull.remote())
    agent.set_weights(weights)

    max_ep_len = env.spec.max_episode_steps

    state = env.reset()
    done = False
    ep_len = 0

    while True:
        if args['render']:
            env.render()
        agent.total_step += 1

        action = agent.select_exploration_action(state)

        next_state, reward, done, _ = env.step(action)
        ep_len += 1

        real_done = False if ep_len >= max_ep_len else done
        mask = 0.0 if real_done else args['gamma']

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
def worker_test(ps, args):

    writer = SummaryWriter('./logs/' + args['algorithm'])

    env = gym.make(args['env_name'])
    env = RescaleAction(env, -1, 1)

    T.manual_seed(args['seed'] * 1000 + 99999)
    np.random.seed(args['seed'] * 1000 + 99999)
    random.seed(args['seed'] * 1000 + 99999)

    env.seed(args['seed'] * 1000 + 99999)
    env.action_space.np_random.seed(args['seed'] * 1000 + 99999)

    best_score = env.reward_range[0]

    agent = Actor(args)

    weights = ray.get(ps.pull.remote())
    agent.set_weights(weights)

    scores = []

    cnt = 0
    while True:

        cnt += 1

        ave_ret = agent._evaluate_agent(env, agent, args)
        scores.append(ave_ret)
        writer.add_scalar('Reward/Test', ave_ret, cnt)

        print("test_reward:", ave_ret)

        if ave_ret > best_score:
            ps.save_weights.remote()
            print("****** weights saved! ******")
            best_score = ave_ret

        np.savetxt("./return.txt", scores, delimiter=",")

        weights = ray.get(ps.pull.remote())
        agent.set_weights(weights)

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
    args = Dict(parser, args.algorithm)

    env = gym.make(args['env_name'])
    env = RescaleAction(env, -1, 1)

    args['n_states'] = env.observation_space.shape[0]
    args['n_actions'] = env.action_space.shape[0]
    args['max_action'] = env.action_space.high[0]
    args['low_action'] = env.action_space.low[0]
    args['max_ep_len'] = env.spec.max_episode_steps

    weights_save_dir = os.path.join(args['save_dir'] + '/' + args['algorithm'] +'/' + args['env_name'], 'sac_weights.pth')

    # Storage location creation
    if not os.path.exists(args['save_dir']):
        os.mkdir(args['save_dir'])

    model_path = args['save_dir'] + '/' + args['algorithm']
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    model_path = model_path + '/' + args['env_name']
    if not os.path.exists(model_path):
        os.mkdir(model_path)


    ray.init()

    if args['restore']:
        ps = ParameterServer.remote([], args, weights_save_dir)
    else:
        net = Learner(args)
        weights = net.get_weights()
        ps = ParameterServer.remote(weights, args, weights_save_dir)

    replay_buffer = ReplayBuffer.remote(args)

    # Start some training tasks.
    for i in range(args['num_workers']):
        worker_rollout.remote(ps, replay_buffer, args, i)

    time.sleep(20)

    for _ in range(args['num_learners']):
        worker_train.remote(ps, replay_buffer, args)

    time.sleep(10)

    task_test = worker_test.remote(ps, args)
    ray.wait([task_test,])
