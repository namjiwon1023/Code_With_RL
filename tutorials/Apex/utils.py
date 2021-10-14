import torch as T
import torch.nn as nn
import numpy as np
import random
import ray
import gym
import time
import os

def reset_parameters(layer, std=1.0, bias_const=1e-6):
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)

class Dict(dict):
    def __init__(self,config, section_name,location = False):
        super(Dict,self).__init__()
        self.initialize(config, section_name,location)
    def initialize(self, config, section_name,location):
        for key,value in config.items(section_name):
            if location :
                self[key] = value
            else:
                self[key] = eval(value)

@ray.remote
class ReplayBuffer:
    def __init__(self, n_states, n_actions, args):

        self.states = np.zeros([args.buffer_size, n_states], dtype=np.float32)
        self.next_states = np.zeros([args.buffer_size, n_states], dtype=np.float32)
        self.actions = np.zeros([args.buffer_size, n_actions], dtype=np.float32)
        self.rewards = np.zeros([args.buffer_size], dtype=np.float32)
        self.masks = np.zeros([args.buffer_size], dtype=np.float32)

        self.max_size = args.buffer_size
        self.ptr, self.cur_len, = 0, 0

    def store(self, state, action, reward, next_state, mask):

        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.masks[self.ptr] = mask

        self.ptr = (self.ptr + 1) % self.max_size
        self.cur_len = min(self.cur_len + 1, self.max_size)

    def sample_batch(self, batch_size):
        index = np.random.choice(self.cur_len, batch_size, replace = False)

        return dict(
                    state = self.states[index],
                    action = self.actions[index],
                    reward = self.rewards[index],
                    next_state = self.next_states[index],
                    mask = self.masks[index],
                    )

    def __len__(self):
        return self.cur_len

    def ready(self, batch_size):
        if self.cur_len >= batch_size:
            return True

def _evaluate_agent(env, agent, args):
    reward_sum = 0
    for _ in range(args.n_starts):
        done = False
        state = env.reset()
        ep_len = 0
        while not (done or (ep_len == args.max_ep_len)):
            ep_len += 1
            if args.render:
                env.render()
            with eval_mode(agent):
                action = agent.select_test_action(state)
            next_state, reward, done, _ = env.step(action)
            reward_sum += reward
            state = next_state
    return reward_sum / args.n_starts

def _random_seed(env, test_env, seed):
    T.manual_seed(seed)
    T.cuda.manual_seed(seed)
    T.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    env.seed(seed)
    env.action_space.np_random.seed(seed)
    test_env.seed(seed+9999)
    test_env.action_space.np_random.seed(seed+9999)

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

@ray.remote
class ParameterServer:
    def __init__(self, weights):
        self.weights = weights

    def push(self, weights):
        self.weights = weights

    def pull(self):
        return self.weights

    def save_weights(self):
        dirPath = os.getcwd() + '/sac_model.pth'
        T.save(self.weights, dirPath)

    def load_weights(self):
        dirPath = os.getcwd() + '/sac_model.pth'
        self.weights = T.load(dirPath)


class Environment:
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.state_dim = self.env.observation_space.shape[0]
        self._max_episode_steps = self.env._max_episode_steps
        self.can_run = False
        self.state = None

        if type(self.env.action_space) == gym.spaces.box.Box : #Continuous
            self.action_dim = self.env.action_space.shape[0]
            self.is_discrete = False
        else :
            self.action_dim = self.env.action_space.n
            self.is_discrete = True

    def reset(self):
        assert not self.can_run
        self.can_run = True
        self.state = self.env.reset()
        return self.state

    def step(self, action):
        assert self.can_run
        next_state, reward, done, info = self.env.step(action)
        self.state = next_state
        if done == True:
            self.can_run = False
        return next_state, reward, done, info


def run_env(env, brain, traj_length = 0, get_traj = False, reward_scaling = 0.1):
    score = 0
    transition = None
    if traj_length == 0:
        traj_length = env._max_episode_steps

    if env.can_run :
        state = env.state
    else :
        state = env.reset()

    for t in range(traj_length):
        if brain.args['value_based'] :
            if brain.args['discrete'] :
                action = brain.get_action(torch.from_numpy(state).float())
                log_prob = np.zeros((1,1))##
            else :
                pass
        else :
            if brain.args['discrete'] :
                prob = brain.get_action(torch.from_numpy(state).float())
                dist = Categorical(prob)
                action = dist.sample()
                log_prob = torch.log(prob.reshape(1,-1).gather(1, action.reshape(1,-1))).detach().cpu().numpy()
                action = action.item()
            else :#continuous
                mu,std = brain.get_action(torch.from_numpy(state).float())
                dist = Normal(mu,std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(-1,keepdim = True).detach().cpu().numpy()
        next_state, reward, done, _ = env.step(action)
        if get_traj :
            transition = make_transition(np.array(state).reshape(1,-1),\
                                        np.array(action).reshape(1,-1),\
                                        np.array(reward * reward_scaling).reshape(1,-1),\
                                        np.array(next_state).reshape(1,-1),\
                                        np.array(float(done)).reshape(1,-1),\
                                        np.array(log_prob))
            brain.put_data(transition)
        score += reward
        if done:
            if not get_traj:
                break
            state = env.reset()
        else :
            state = next_state
    return score

