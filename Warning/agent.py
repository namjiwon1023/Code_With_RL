import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import copy
import gym
import time

from network import QNetwork
from replaybuffer import ReplayBuffer

from utils import _target_net_update, _target_soft_update, compute_gae, ppo_iter, _load_model

class DQNAgent(object):
    def __init__(self, args):

        self.args = args

        self.env = gym.make(args.env_name)
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n

        self.eval = QNetwork(self.n_states, self.n_actions, args)

        self.target = copy.deepcopy(self.eval)
        self.target.eval()
        for p in self.target.parameters():
            p.requires_grad = False

        self.optimizer = optim.Adam(self.eval.parameters(), lr=self.args.critic_lr)

        self.memory = ReplayBuffer(self.n_states, self.n_actions, args)
        self.transition = list()

        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)

        self.model_path = self.args.save_dir + '/' + args.algorithm
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        self.model_path = self.model_path + '/' + args.env_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        if os.path.exists(self.model_path + '/DQN.pth'):
            _load_model(self.eval)

        self.total_step = 0

    def choose_action(self, state, epsilon):
        with T.no_grad():
            if epsilon >= np.random.random() and not self.args.evaluate:
                choose_action = np.random.randint(0, self.n_actions)
            else :
                choose_action = self.eval(T.as_tensor(state, dtype=T.float32, device=self.args.device)).argmax()
                choose_action = choose_action.detach().cpu().numpy()

            if not self.args.evaluate:
                self.transition = [state, choose_action]
        return choose_action

    def learn(self):
        Q_loss = 0

        with T.no_grad():
            samples = self.memory.sample_batch(self.args.batch_size)

            state = T.as_tensor(samples['state'], dtype=T.float32, device=self.args.device)
            next_state = T.as_tensor(samples['next_state'], dtype=T.float32, device=self.args.device)
            action = T.as_tensor(samples['action'], dtype=T.long, device=self.args.device).reshape(-1, self.n_actions)
            reward = T.as_tensor(samples['reward'], dtype=T.float32, device=self.args.device).reshape(-1,1)
            mask = T.as_tensor(samples['mask'], dtype=T.float32, device=self.args.device).reshape(-1,1)

            next_q = self.target(next_state).max(dim=1, keepdim=True)[0]
            target_q = reward + next_q * mask

        curr_q = self.eval(state).gather(1, action)

        loss = (target_q - curr_q)**2
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        Q_loss += loss.detach().item()

        if self.total_step % self.args.update_rate == 0:
            _target_net_update(self.eval, self.target)

        return Q_loss
