# Copyright (c) 2021: Zhiyuan Nan (namjw@hanyang.ac.kr).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical
import numpy as np
import os
import copy
import gym
from gym.wrappers import RescaleAction

from collections import deque
from test.network import QNetwork, DuelingNetwork
from test.network import Actor, ActorA2C, ActorPPO, ActorSAC, CriticQ, CriticV, CriticTwin
from test.replaybuffer import ReplayBuffer, ReplayBufferPPO
from test.utils import OUNoise

class DQNAgent(object):
    def __init__(self, args):

        self.args = args

        # Environment setting
        self.env = gym.make(args.env_name)
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n

        self.checkpoint = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, 'DQN.pth')

        # network setting
        self.eval = QNetwork(self.n_states, self.n_actions, args)
        self.criterion = nn.MSELoss()

        self.target = copy.deepcopy(self.eval)
        self.target.eval()
        for p in self.target.parameters():
            p.requires_grad = False

        # optimizer setting
        self.optimizer = optim.Adam(self.eval.parameters(), lr=self.args.critic_lr)

        # replay buffer
        self.memory = ReplayBuffer(self.n_states, self.n_actions, args)
        self.transition = list()

        # Storage location creation
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)

        self.model_path = self.args.save_dir + '/' + args.algorithm
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        self.model_path = self.model_path + '/' + args.env_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        if os.path.exists(self.model_path + '/DQN.pth'):
            self.load_models()

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

        # TD error
        critic_loss = self._value_update(self.memory, self.args.batch_size)

        # update value
        self.optimizer.zero_grad()
        critic_loss.backward()
        self.optimizer.step()

        # target network hard update
        if self.total_step % self.args.update_rate == 0:
            self._target_net_update(self.target, self.eval)

    def save_models(self):
        print('------ Save model ------')
        _save_model(self.eval, self.checkpoint)

    def load_models(self):
        print('------ load model ------')
        _load_model(self.eval, self.checkpoint)

    # target network hard update
    def _target_net_update(self, target_net, eval_net):
        target_net.load_state_dict(eval_net.state_dict())

    def _value_update(self, buffer, batch_size):
        with T.no_grad():
            # Select data from ReplayBuffer with batch_size size
            samples = buffer.sample_batch(batch_size)
            state = T.as_tensor(samples['state'], dtype=T.float32, device=self.args.device)
            next_state = T.as_tensor(samples['next_state'], dtype=T.float32, device=self.args.device)
            action = T.as_tensor(samples['action'], dtype=T.long, device=self.args.device).reshape(-1, 1)
            reward = T.as_tensor(samples['reward'], dtype=T.float32, device=self.args.device).reshape(-1,1)
            mask = T.as_tensor(samples['mask'], dtype=T.float32, device=self.args.device).reshape(-1,1)

            next_q = self.target(next_state).max(dim=1, keepdim=True)[0]
            # Here we calculate action value Q(s,a) = R + yV(s')
            target_q = reward + next_q * mask

        current_q = self.eval(state).gather(1, action)
        loss = self.criterion(current_q, target_q)

        return loss

class DoubleDQNAgent(object):
    def __init__(self, args):

        self.args = args

        # Environment setting
        self.env = gym.make(args.env_name)
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n

        self.checkpoint = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, 'DoubleDQN.pth')

        # network setting
        self.eval = QNetwork(self.n_states, self.n_actions, args)
        self.criterion = nn.MSELoss()

        self.target = copy.deepcopy(self.eval)
        self.target.eval()
        for p in self.target.parameters():
            p.requires_grad = False

        # optimizer setting
        self.optimizer = optim.Adam(self.eval.parameters(), lr=self.args.critic_lr)

        # replay buffer
        self.memory = ReplayBuffer(self.n_states, self.n_actions, args)
        self.transition = list()

        # Storage location creation
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)

        self.model_path = self.args.save_dir + '/' + args.algorithm
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        self.model_path = self.model_path + '/' + args.env_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        if os.path.exists(self.model_path + '/DoubleDQN.pth'):
            self.load_models()

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

        # TD error
        critic_loss = self._value_update(self.memory, self.args.batch_size)

        # update value
        self.optimizer.zero_grad()
        critic_loss.backward()
        self.optimizer.step()

        # target network hard update
        if self.total_step % self.args.update_rate == 0:
            self._target_net_update(self.target, self.eval)

    def save_models(self):
        print('------ Save model ------')
        _save_model(self.eval, self.checkpoint)

    def load_models(self):
        print('------ load model ------')
        _load_model(self.eval, self.checkpoint)

    # target network hard update
    def _target_net_update(self, target_net, eval_net):
        target_net.load_state_dict(eval_net.state_dict())

    def _value_update(self, buffer, batch_size):
        with T.no_grad():
            # Select data from ReplayBuffer with batch_size size
            samples = buffer.sample_batch(batch_size)
            state = T.as_tensor(samples['state'], dtype=T.float32, device=self.args.device)
            next_state = T.as_tensor(samples['next_state'], dtype=T.float32, device=self.args.device)
            action = T.as_tensor(samples['action'], dtype=T.long, device=self.args.device).reshape(-1, 1)
            reward = T.as_tensor(samples['reward'], dtype=T.float32, device=self.args.device).reshape(-1,1)
            mask = T.as_tensor(samples['mask'], dtype=T.float32, device=self.args.device).reshape(-1,1)

            # Double DQN
            next_q = self.target(next_state).gather(1, self.eval(next_state).argmax(dim = 1, keepdim = True))
            # Here we calculate action value Q(s,a) = R + yV(s')
            target_q = reward + next_q * mask

        current_q = self.eval(state).gather(1, action)
        # TD error
        loss = self.criterion(current_q, target_q)

        return loss

class DuelingDQNAgent(object):
    def __init__(self, args):

        self.args = args

        # Environment setting
        self.env = gym.make(args.env_name)
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n

        self.checkpoint = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, 'DuelingDQN.pth')

        # network setting
        self.eval = DuelingNetwork(self.n_states, self.n_actions, args)

        # loss function
        self.criterion = nn.MSELoss()

        self.target = copy.deepcopy(self.eval)
        self.target.eval()
        for p in self.target.parameters():
            p.requires_grad = False

        # optimizer setting
        self.optimizer = optim.Adam(self.eval.parameters(), lr=self.args.critic_lr)

        # replay buffer
        self.memory = ReplayBuffer(self.n_states, self.n_actions, args)
        self.transition = list()

        # Storage location creation
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)

        self.model_path = self.args.save_dir + '/' + args.algorithm
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        self.model_path = self.model_path + '/' + args.env_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        if os.path.exists(self.model_path + '/DuelingDQN.pth'):
            self.load_models()

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
        # TD error
        critic_loss = self._value_update(self.memory, self.args.batch_size)

        # update value
        self.optimizer.zero_grad()
        critic_loss.backward()
        self.optimizer.step()

        # target network hard update
        if self.total_step % self.args.update_rate == 0:
            self._target_net_update(self.target, self.eval)

    def save_models(self):
        print('------ Save model ------')
        _save_model(self.eval, self.checkpoint)

    def load_models(self):
        print('------ load model ------')
        _load_model(self.eval, self.checkpoint)

    # target network hard update
    def _target_net_update(self, target_net, eval_net):
        target_net.load_state_dict(eval_net.state_dict())

    def _value_update(self, buffer, batch_size):
        with T.no_grad():
            # Select data from ReplayBuffer with batch_size size
            samples = buffer.sample_batch(batch_size)
            state = T.as_tensor(samples['state'], dtype=T.float32, device=self.args.device)
            next_state = T.as_tensor(samples['next_state'], dtype=T.float32, device=self.args.device)
            action = T.as_tensor(samples['action'], dtype=T.long, device=self.args.device).reshape(-1, 1)
            reward = T.as_tensor(samples['reward'], dtype=T.float32, device=self.args.device).reshape(-1,1)
            mask = T.as_tensor(samples['mask'], dtype=T.float32, device=self.args.device).reshape(-1,1)

            next_q = self.target(next_state).max(dim=1, keepdim=True)[0]
            # Here we calculate action value Q(s,a) = R + yV(s')
            target_q = reward + next_q * mask

        current_q = self.eval(state).gather(1, action)
        loss = self.criterion(current_q, target_q)

        return loss

class D3QNAgent(object):
    def __init__(self, args):

        self.args = args

        # Environment setting
        self.env = gym.make(args.env_name)
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n

        self.checkpoint = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, 'D3QN.pth')

        # network setting
        self.eval = DuelingNetwork(self.n_states, self.n_actions, args)

        # loss function
        self.criterion = nn.MSELoss()

        self.target = copy.deepcopy(self.eval)
        self.target.eval()
        for p in self.target.parameters():
            p.requires_grad = False

        # optimizer setting
        self.optimizer = optim.Adam(self.eval.parameters(), lr=self.args.critic_lr)

        # replay buffer
        self.memory = ReplayBuffer(self.n_states, self.n_actions, args)
        self.transition = list()

        # Storage location creation
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)

        self.model_path = self.args.save_dir + '/' + args.algorithm
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        self.model_path = self.model_path + '/' + args.env_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        if os.path.exists(self.model_path + '/D3QN.pth'):
            self.load_models()

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
        # TD error
        critic_loss = self._value_update(self.memory, self.args.batch_size)

        # update value
        self.optimizer.zero_grad()
        critic_loss.backward()
        self.optimizer.step()

        # target network hard update
        if self.total_step % self.args.update_rate == 0:
            self._target_net_update(self.target, self.eval)


    def save_models(self):
        print('------ Save model ------')
        _save_model(self.eval, self.checkpoint)

    def load_models(self):
        print('------ load model ------')
        _load_model(self.eval, self.checkpoint)

    # target network hard update
    def _target_net_update(self, target_net, eval_net):
        target_net.load_state_dict(eval_net.state_dict())

    def _value_update(self, buffer, batch_size):
        with T.no_grad():
            # Select data from ReplayBuffer with batch_size size
            samples = buffer.sample_batch(batch_size)
            state = T.as_tensor(samples['state'], dtype=T.float32, device=self.args.device)
            next_state = T.as_tensor(samples['next_state'], dtype=T.float32, device=self.args.device)
            action = T.as_tensor(samples['action'], dtype=T.long, device=self.args.device).reshape(-1, 1)
            reward = T.as_tensor(samples['reward'], dtype=T.float32, device=self.args.device).reshape(-1,1)
            mask = T.as_tensor(samples['mask'], dtype=T.float32, device=self.args.device).reshape(-1,1)

            # Double DQN
            next_q = self.target(next_state).gather(1, self.eval(next_state).argmax(dim = 1, keepdim = True))
            # Here we calculate action value Q(s,a) = R + yV(s')
            target_q = reward + next_q * mask

        current_q = self.eval(state).gather(1, action)
        # TD error
        loss = self.criterion(current_q, target_q)

        return loss

class NoisyDQNAgent(object):
    def __init__(self, args):

        self.args = args

        # Environment setting
        self.env = gym.make(args.env_name)
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n

        self.checkpoint = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, 'NoisyDQN.pth')

        # network setting
        self.eval = QNetwork(self.n_states, self.n_actions, args)

        # loss function
        self.criterion = nn.MSELoss()

        self.target = copy.deepcopy(self.eval)
        self.target.eval()
        for p in self.target.parameters():
            p.requires_grad = False

        # optimizer setting
        self.optimizer = optim.Adam(self.eval.parameters(), lr=self.args.critic_lr)

        # replay buffer
        self.memory = ReplayBuffer(self.n_states, self.n_actions, args)
        self.transition = list()

        # Storage location creation
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)

        self.model_path = self.args.save_dir + '/' + args.algorithm
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        self.model_path = self.model_path + '/' + args.env_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        if os.path.exists(self.model_path + '/NoisyDQN.pth'):
            self.load_models()

        self.total_step = 0

    def choose_action(self, state):
        with T.no_grad():
            choose_action = self.eval(T.as_tensor(state, dtype=T.float32, device=self.args.device)).argmax()
            choose_action = choose_action.detach().cpu().numpy()
            if not self.args.evaluate:
                self.transition = [state, choose_action]
        return choose_action

    def learn(self):
        # TD error
        critic_loss = self._value_update(self.memory, self.args.batch_size)

        # update value
        self.optimizer.zero_grad()
        critic_loss.backward()
        self.optimizer.step()

        # noise update
        self.eval.reset_noise()
        self.target.reset_noise()

        # target network hard update
        if self.total_step % self.args.update_rate == 0:
            self._target_net_update(self.target, self.eval)

    def save_models(self):
        print('------ Save model ------')
        _save_model(self.eval, self.checkpoint)

    def load_models(self):
        print('------ load model ------')
        _load_model(self.eval, self.checkpoint)

    # target network hard update
    def _target_net_update(self, target_net, eval_net):
        target_net.load_state_dict(eval_net.state_dict())

    def _value_update(self, buffer, batch_size):
        with T.no_grad():
            # Select data from ReplayBuffer with batch_size size
            samples = buffer.sample_batch(batch_size)
            state = T.as_tensor(samples['state'], dtype=T.float32, device=self.args.device)
            next_state = T.as_tensor(samples['next_state'], dtype=T.float32, device=self.args.device)
            action = T.as_tensor(samples['action'], dtype=T.long, device=self.args.device).reshape(-1, 1)
            reward = T.as_tensor(samples['reward'], dtype=T.float32, device=self.args.device).reshape(-1,1)
            mask = T.as_tensor(samples['mask'], dtype=T.float32, device=self.args.device).reshape(-1,1)

            next_q = self.target(next_state).max(dim=1, keepdim=True)[0]
            # Here we calculate action value Q(s,a) = R + yV(s')
            target_q = reward + next_q * mask

        current_q = self.eval(state).gather(1, action)
        loss = self.criterion(current_q, target_q)

        return loss

class DDPGAgent(object):
    def __init__(self, args):
        self.args = args
        self.actor_path = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, 'ddpg_actor.pth')
        self.critic_path = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, 'ddpg_critic.pth')

        # Environment setting
        self.env = gym.make(args.env_name)
        self.env = RescaleAction(self.env, -1, 1)
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]

        self.max_action = self.env.action_space.high[0]
        self.low_action = self.env.action_space.low[0]

        # OU noise setting
        self.noise = OUNoise(self.n_actions, theta=self.args.ou_noise_theta, sigma=self.args.ou_noise_sigma,)

        # replay buffer
        self.memory = ReplayBuffer(self.n_states, self.n_actions, self.args)
        self.transition = list()

        # actor-critic network setting
        self.actor_eval = Actor(self.n_states, self.n_actions, self.args)
        self.critic_eval = CriticQ(self.n_states, self.n_actions, self.args)

        # optimizer setting
        self.actor_optimizer = optim.Adam(self.actor_eval.parameters(), lr=self.args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_eval.parameters(), lr=self.args.critic_lr)

        # loss function
        self.criterion = nn.MSELoss()

        self.actor_target = copy.deepcopy(self.actor_eval)
        self.actor_target.eval()
        for p in self.actor_target.parameters():
            p.requires_grad = False

        self.critic_target = copy.deepcopy(self.critic_eval)
        self.critic_target.eval()
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # Storage location creation
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)

        self.model_path = self.args.save_dir + '/' + args.algorithm
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        self.model_path = self.model_path + '/' + args.env_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        if os.path.exists(self.model_path + '/ddpg_actor.pth'):
            self.load_models()

        self.total_step = 0

    def choose_action(self, state, epsilon):
        with T.no_grad():
            if epsilon > np.random.random() and not self.args.evaluate:
                choose_action = np.random.uniform(self.low_action, self.max_action, self.n_actions)
            else :
                choose_action = self.actor_eval(T.as_tensor(state, dtype=T.float32, device=self.args.device)).detach().cpu().numpy()

            if not self.args.evaluate:
                if self.args.Gaussian_noise:
                    noise = np.random.normal(0, self.max_action*self.args.exploration_noise, size=self.n_actions)
                else:
                    noise = self.noise.sample()
                choose_action = np.clip(choose_action + noise, self.low_action, self.max_action)
            self.transition = [state, choose_action]
        return choose_action

    def learn(self):
        # TD error
        critic_loss, state = self._value_update(self.memory, self.args.batch_size)

        # update value
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # critic target network soft update
        if self.total_step % self.args.target_update_interval == 0:
            self._target_soft_update(self.critic_target, self.critic_eval, self.args.tau)

        for p in self.critic_eval.parameters():
            p.requires_grad = False

        # actor network loss function
        actor_loss = self._policy_update(state)

        # update policy
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # actor target network soft update
        if self.total_step % self.args.target_update_interval == 0:
            self._target_soft_update(self.actor_target, self.actor_eval, self.args.tau)

        for p in self.critic_eval.parameters():
            p.requires_grad = True

    def save_models(self):
        print('------ Save model ------')
        _save_model(self.actor_eval, self.actor_path)
        _save_model(self.critic_eval, self.critic_path)

    def load_models(self):
        print('------ load model ------')
        _load_model(self.actor_eval, self.actor_path)
        _load_model(self.critic_eval, self.critic_path)

    # target network soft update
    def _target_soft_update(self, target_net, eval_net , tau=None):
        if tau == None:
            tau = self.args.tau
        with T.no_grad():
            for t_p, l_p in zip(target_net.parameters(), eval_net.parameters()):
                t_p.data.copy_(tau * l_p.data + (1 - tau) * t_p.data)

    def _value_update(self, buffer, batch_size):
        with T.no_grad():
            # Select data from ReplayBuffer with batch_size size
            samples = buffer.sample_batch(batch_size)

            state = T.as_tensor(samples['state'], dtype=T.float32, device=self.args.device)
            next_state = T.as_tensor(samples['next_state'], dtype=T.float32, device=self.args.device)
            action = T.as_tensor(samples['action'], dtype=T.float32, device=self.args.device).reshape(-1, 1)
            reward = T.as_tensor(samples['reward'], dtype=T.float32, device=self.args.device).reshape(-1,1)
            mask = T.as_tensor(samples['mask'], dtype=T.float32, device=self.args.device).reshape(-1,1)

            next_action = self.actor_target(next_state)
            next_value = self.critic_target(next_state, next_action)
            # Here we calculate action value Q(s,a) = R + yV(s')
            target_values = reward + next_value * mask

        eval_values = self.critic_eval(state, action)
        # TD error
        critic_loss = self.criterion(eval_values, target_values)

        return critic_loss, state

    def _policy_update(self, state):
        actor_loss = -self.critic_eval(state, self.actor_eval(state)).mean()
        return actor_loss

class TD3Agent(object):
    def __init__(self, args):
        self.args = args
        self.actor_path = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, 'td3_actor.pth')
        self.critic_path = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, 'td3_critic.pth')

        # Environment setting
        self.env = gym.make(args.env_name)
        self.env = RescaleAction(self.env, -1, 1)
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]

        self.max_action = self.env.action_space.high[0]
        self.low_action = self.env.action_space.low[0]

        # replay buffer
        self.memory = ReplayBuffer(self.n_states, self.n_actions, self.args)
        self.transition = list()

        # actor-critic net setting
        self.actor_eval = Actor(self.n_states, self.n_actions, self.args)
        self.critic_eval = CriticTwin(self.n_states, self.n_actions, self.args)

        # loss function
        self.criterion = nn.MSELoss()

        # optimizer setting
        self.actor_optimizer = optim.Adam(self.actor_eval.parameters(), lr=self.args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_eval.parameters(), lr=self.args.critic_lr)

        self.actor_target = copy.deepcopy(self.actor_eval)
        self.actor_target.eval()
        for p in self.actor_target.parameters():
            p.requires_grad = False

        self.critic_target = copy.deepcopy(self.critic_eval)
        self.critic_target.eval()
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # Storage location creation
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)

        self.model_path = self.args.save_dir + '/' + args.algorithm
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        self.model_path = self.model_path + '/' + args.env_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        if os.path.exists(self.model_path + '/td3_actor.pth'):
            self.load_models()

        self.total_step = 0

    def choose_action(self, state, epsilon):
        with T.no_grad():
            if epsilon >= np.random.random() and not self.args.evaluate:
                choose_action = np.random.uniform(self.low_action, self.max_action, self.n_actions)
            else :
                choose_action = self.actor_eval(T.as_tensor(state, device=self.actor_eval.device, dtype=T.float32)).detach().cpu().numpy()
            if not self.args.evaluate:
                noise = np.random.normal(0, self.max_action*self.args.exploration_noise, size=self.n_actions)
                choose_action = np.clip(choose_action + noise, self.low_action, self.max_action)
            self.transition = [state, choose_action]
        return choose_action

    def learn(self):
        q1_loss, q2_loss, state = self._value_update(self.memory, self.args.batch_size)
        critic_loss = q1_loss + q2_loss

        # update value
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_step % self.args.policy_freq == 0:
            self._target_soft_update(self.critic_target, self.critic_eval, self.args.tau)

        for p in self.critic_eval.parameters():
            p.requires_grad = False

        # actor network delay update
        if self.total_step % self.args.policy_freq == 0:
            # actor loss function
            actor_loss = self._policy_update(state)

            # update policy
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # target network soft update
            self._target_soft_update(self.actor_target, self.actor_eval, self.args.tau)

        for p in self.critic_eval.parameters():
            p.requires_grad = True

    def save_models(self):
        print('------ Save model ------')
        _save_model(self.actor_eval, self.actor_path)
        _save_model(self.critic_eval, self.critic_path)

    def load_models(self):
        print('------ load model ------')
        _load_model(self.actor_eval, self.actor_path)
        _load_model(self.critic_eval, self.critic_path)

    # target network soft update
    def _target_soft_update(self, target_net, eval_net , tau=None):
        if tau == None:
            tau = self.args.tau
        with T.no_grad():
            for t_p, l_p in zip(target_net.parameters(), eval_net.parameters()):
                t_p.data.copy_(tau * l_p.data + (1 - tau) * t_p.data)

    def _value_update(self, buffer, batch_size):
        with T.no_grad():
            # Select data from ReplayBuffer with batch_size size
            samples = buffer.sample_batch(batch_size)

            state = T.as_tensor(samples['state'], dtype=T.float32, device=self.args.device)
            next_state = T.as_tensor(samples['next_state'], dtype=T.float32, device=self.args.device)
            action = T.as_tensor(samples['action'], dtype=T.float32, device=self.args.device).reshape(-1, 1)
            reward = T.as_tensor(samples['reward'], dtype=T.float32, device=self.args.device).reshape(-1,1)
            mask = T.as_tensor(samples['mask'], dtype=T.float32, device=self.args.device).reshape(-1,1)

            noise = (T.randn_like(action) * self.args.policy_noise * self.max_action).clamp(self.args.noise_clip * self.low_action, self.args.noise_clip * self.max_action)
            next_action = (self.actor_target(next_state) + noise).clamp(self.low_action, self.max_action)

            next_target_q1, next_target_q2 = self.critic_target.get_double_q(next_state, next_action)
            next_target_q = T.min(next_target_q1, next_target_q2)
            # Here we calculate action value Q(s,a) = R + yV(s')
            target_q = reward + next_target_q * mask

        # Twin Critic Network Loss functions
        current_q1, current_q2 = self.critic_eval.get_double_q(state, action)
        # TD error
        q1_loss = self.criterion(current_q1, target_q)
        q2_loss = self.criterion(current_q2, target_q)

        return q1_loss, q2_loss, state

    def _policy_update(self, state):
        actor_loss = -self.critic_eval(state, self.actor_eval(state)).mean()
        return actor_loss

class SACAgent(object):
    def __init__(self, args):
        self.args = args
        self.actor_path = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, 'sac_actor.pth')
        self.critic_path = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, 'sac_critic.pth')

        # Environment setting
        self.env = gym.make(args.env_name)
        self.env = RescaleAction(self.env, -1, 1)
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]

        self.max_action = self.env.action_space.high[0]
        self.low_action = self.env.action_space.low[0]

        # replay buffer
        self.memory = ReplayBuffer(self.n_states, self.n_actions, self.args)
        self.transition = list()

        # actor-critic net setting
        self.actor = ActorSAC(self.n_states, self.n_actions, self.args)
        self.critic_eval = CriticTwin(self.n_states, self.n_actions, self.args)

        # loss function
        self.criterion = nn.MSELoss()

        # optimizer setting
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_eval.parameters(), lr=self.args.critic_lr)

        self.critic_target = copy.deepcopy(self.critic_eval)
        self.critic_target.eval()
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # Temperature Coefficient
        self.target_entropy = -self.n_actions
        self.log_alpha = T.zeros(1, requires_grad=True, device=self.args.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.args.alpha_lr)

        # Storage location creation
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)

        self.model_path = self.args.save_dir + '/' + args.algorithm
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        self.model_path = self.model_path + '/' + args.env_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        if os.path.exists(self.model_path + '/sac_actor.pth'):
            self.load_models()

        self.total_step = 0

    def choose_action(self, state, epsilon):
        with T.no_grad():
            if epsilon >= np.random.random() and not self.args.evaluate:
                choose_action = np.random.uniform(self.low_action, self.max_action, self.n_actions)
            else :
                choose_action, _ = self.actor(T.as_tensor(state, dtype=T.float32, device=self.actor.device))
                choose_action = choose_action.detach().cpu().numpy()
            self.transition = [state, choose_action]
        return choose_action

    def learn(self):
        # TD error
        # update value
        q1_loss, q2_loss, state = self._value_update(self.memory, self.args.batch_size)
        critic_loss = q1_loss + q2_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # target network soft update
        if self.total_step % self.args.target_update_interval == 0:
            self._target_soft_update(self.critic_target, self.critic_eval, self.args.tau)

        for p in self.critic_eval.parameters():
            p.requires_grad = False

        actor_loss, new_log_prob = self._policy_update(state)

        # update Policy
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update Temperature Coefficient
        alpha_loss = self._temperature_update(new_log_prob)

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        for p in self.critic_eval.parameters():
            p.requires_grad = True

        self.alpha = self.log_alpha.exp()

    def save_models(self):
        print('------ Save model ------')
        _save_model(self.actor, self.actor_path)
        _save_model(self.critic_eval, self.critic_path)
        checkpoint = os.path.join(self.args.save_dir + '/' + self.args.algorithm +'/' + self.args.env_name, 'log_alpha.pth')
        T.save(self.log_alpha, checkpoint)

    def load_models(self):
        print('------ load model ------')
        _load_model(self.actor, self.actor_path)
        _load_model(self.critic_eval, self.critic_path)
        checkpoint = os.path.join(self.args.save_dir + '/' + self.args.algorithm +'/' + self.args.env_name, 'log_alpha.pth')
        self.log_alpha = T.load(checkpoint)

    # target network soft update
    def _target_soft_update(self, target_net, eval_net , tau=None):
        if tau == None:
            tau = self.args.tau
        with T.no_grad():
            for t_p, l_p in zip(target_net.parameters(), eval_net.parameters()):
                t_p.data.copy_(tau * l_p.data + (1 - tau) * t_p.data)

    def _value_update(self, buffer, batch_size):
        with T.no_grad():
            # Select data from ReplayBuffer with batch_size size
            samples = buffer.sample_batch(batch_size)

            state = T.as_tensor(samples['state'], dtype=T.float32, device=self.args.device)
            next_state = T.as_tensor(samples['next_state'], dtype=T.float32, device=self.args.device)
            action = T.as_tensor(samples['action'], dtype=T.float32, device=self.args.device).reshape(-1, 1)
            reward = T.as_tensor(samples['reward'], dtype=T.float32, device=self.args.device).reshape(-1,1)
            mask = T.as_tensor(samples['mask'], dtype=T.float32, device=self.args.device).reshape(-1,1)

            next_action, next_log_prob = self.actor(next_state)
            next_target_q1, next_target_q2 = self.critic_target.get_double_q(next_state, next_action)
            next_target_q = T.min(next_target_q1, next_target_q2)
            # Here we calculate action value Q(s,a) = R + yV(s')
            target_q = reward + (next_target_q - self.alpha * next_log_prob) * mask

        # Twin Critic Network Loss functions
        current_q1, current_q2 = self.critic_eval.get_double_q(state, action)
        # TD error
        # update value
        q1_loss = self.criterion(current_q1, target_q)
        q2_loss = self.criterion(current_q2, target_q)
        return q1_loss, q2_loss, state

    def _policy_update(self, state):
        new_action, new_log_prob = self.actor(state)
        q_1, q_2 = self.critic_eval.get_double_q(state, new_action)
        q = T.min(q_1, q_2)
        # update actor network
        actor_loss = (self.alpha * new_log_prob - q).mean()
        return actor_loss, new_log_prob

    def _temperature_update(self, new_log_prob):
        alpha_loss = -self.log_alpha * (new_log_prob.detach() + self.target_entropy).mean()
        return alpha_loss

class PPOAgent(object):
    def __init__(self, args):
        self.args = args
        self.actor_path = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, 'ppo_actor.pth')
        self.critic_path = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, 'ppo_critic.pth')

        # Environment setting
        self.env = gym.make(args.env_name)

        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]

        self.max_action = self.env.action_space.high[0]
        self.low_action = self.env.action_space.low[0]

        # actor-critic net setting
        self.actor = ActorPPO(self.n_states, self.n_actions, self.args)
        self.critic = CriticV(self.n_states, self.args)

        # loss function
        self.criterion = nn.MSELoss()

        # optimizer setting
        self.optimizer = optim.Adam([{'params': self.actor.parameters(), 'lr': self.args.actor_lr},
                                    {'params': self.critic.parameters(), 'lr': self.args.critic_lr}])


        # Storage location creation
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)

        self.model_path = self.args.save_dir + '/' + args.algorithm
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        self.model_path = self.model_path + '/' + args.env_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        if os.path.exists(self.model_path + '/ppo_actor.pth'):
            self.load_models()

        self.total_step = 0

        # simple replay buffer
        self.memory = ReplayBufferPPO()

    def choose_action(self, state):
        state = T.as_tensor((state,), dtype=T.float32, device=self.args.device)
        mu, std = self.actor(state)
        if self.args.evaluate and not self.args.is_discrete:
            choose_action = mu
        if not self.args.evaluate:
            value = self.critic(state)
            self.memory.values.append(value)
            self.memory.states.append(state)
            dist = Normal(mu, std)
            choose_action = dist.sample()
            self.memory.actions.append(choose_action)
            self.memory.log_probs.append(dist.log_prob(choose_action))
        return choose_action.detach().cpu().numpy()[0]

    def learn(self, next_state):
        next_state = T.as_tensor((next_state,), dtype=T.float32, device=self.args.device)
        next_value = self.critic(next_state)

        returns = compute_gae(next_value, self.memory.rewards, self.memory.masks, self.memory.values, self.args.gamma, self.args.tau)

        states = T.cat(self.memory.states)
        actions = T.cat(self.memory.actions)
        returns = T.cat(returns).detach()
        values = T.cat(self.memory.values).detach()
        log_probs = T.cat(self.memory.log_probs).detach()

        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advantages = returns - values

        if self.args.is_discrete:
            actions = actions.unsqueeze(1)
            log_probs = log_probs.unsqueeze(1)

        # Normalize the advantages
        if self.args.standardize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        for state, action, old_value, old_log_prob, return_, adv in ppo_iter(epoch = self.args.epoch,
                                                                            mini_batch_size = self.args.batch_size,
                                                                            states = states,
                                                                            actions = actions,
                                                                            values = values,
                                                                            log_probs = log_probs,
                                                                            returns = returns,
                                                                            advantages = advantages,
                                                                            ):
            mu, std = self.actor(state)
            dist = Normal(mu, std)

            entropy = dist.entropy().mean()
            log_prob = dist.log_prob(action)

            ratio = (log_prob - old_log_prob).exp()

            surr1 = ratio * adv
            surr2 = T.clamp(ratio, 1.0 - self.args.epsilon, 1.0 + self.args.epsilon) * adv

            # actor network loss function
            # update policy
            actor_loss  = - T.min(surr1, surr2).mean()

            value = self.critic(state)

            # TD error
            # update value
            if self.args.use_clipped_value_loss:
                value_pred_clipped = old_value + T.clamp((value - old_value), - self.args.epsilon, self.args.epsilon)
                value_loss_clipped = (return_ - value_pred_clipped).pow(2)
                value_loss = (return_ - value).pow(2)
                critic_loss = T.max(value_loss, value_loss_clipped).mean()
            else:
                critic_loss = self.criterion(value, return_)

            # PPO total loss function
            total_loss = self.args.value_weight * critic_loss + actor_loss - entropy * self.args.entropy_weight

            # Policy gradient update
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        # After the update, clear the memory
        self.memory.RB_clear()


    def save_models(self):
        print('------ Save model ------')
        _save_model(self.actor, self.actor_path)
        _save_model(self.critic, self.critic_path)

    def load_models(self):
        print('------ load model ------')
        _load_model(self.actor, self.actor_path)
        _load_model(self.critic, self.critic_path)

class A2CAgent(object):
    def __init__(self, args):
        self.args = args
        self.actor_path = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, 'a2c_actor.pth')
        self.critic_path = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, 'a2c_critic.pth')

        # Environment setting
        self.env = gym.make(args.env_name)

        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]

        self.max_action = self.env.action_space.high[0]
        self.low_action = self.env.action_space.low[0]

        self.transition = list()

        # actor-critic net setting
        self.actor = ActorA2C(self.n_states, self.n_actions, self.args)
        self.critic = CriticV(self.n_states, self.args)

        # Loss Function
        self.criterion = nn.SmoothL1Loss()

        # optimizer setting
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.args.critic_lr)

        # Storage location creation
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)

        self.model_path = self.args.save_dir + '/' + args.algorithm
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        self.model_path = self.model_path + '/' + args.env_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        if os.path.exists(self.model_path + '/a2c_actor.pth'):
            self.load_models()

        self.total_step = 0

    def choose_action(self, state):
        state = T.as_tensor(state, dtype=T.float32, device=self.args.device)
        mu, std = self.actor(state)
        if self.args.evaluate:
            choose_action = mu
        else:
            dist = Normal(mu, std)
            choose_action = dist.sample()
            log_prob = dist.log_prob(choose_action).sum(dim=-1)
            self.transition = [state, log_prob]
        return choose_action.clamp(self.low_action, self.max_action).detach().cpu().numpy()

    def learn(self):
        critic_loss, target_value, current_value, log_prob = self._value_update(self.transition)

        # update value
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = self._policy_update(target_value, current_value, log_prob, self.args.entropy_weight)

        # update policy
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def save_models(self):
        print('------ Save model ------')
        _save_model(self.actor, self.actor_path)
        _save_model(self.critic, self.critic_path)

    def load_models(self):
        print('------ load model ------')
        _load_model(self.actor, self.actor_path)
        _load_model(self.critic, self.critic_path)

    def _value_update(self, transition):
        state, log_prob, reward, next_state, mask = transition
        # Q(s,a)   = r + gamma * V(s`)  if state != Terminal
        #       = r                       otherwise
        next_state = T.as_tensor(next_state, dtype=T.float32, device=self.args.device)
        current_value = self.critic(state)
        next_value = self.critic(next_state).detach()
        target_value = reward + next_value * mask
        critic_loss = self.criterion(current_value, target_value)
        return critic_loss, target_value, current_value, log_prob

    def _policy_update(self, target_value, current_value, log_prob, entropy_weight):
        # advantage = Q_t - V(s_t)
        advantage = (target_value - current_value).detach()  # not backpropagated
        actor_loss = -advantage * log_prob
        actor_loss += entropy_weight * -log_prob  # entropy maximization
        return actor_loss

class BC_SACAgent(object):
    def __init__(self, args):
        self.args = args
        self.actor_path = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, 'sac_actor.pth')
        self.critic_path = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, 'sac_critic.pth')

        # Environment setting
        self.env = gym.make(args.env_name)
        self.env = RescaleAction(self.env, -1, 1)
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]

        self.max_action = self.env.action_space.high[0]
        self.low_action = self.env.action_space.low[0]

        # replay buffer
        self.memory = ReplayBuffer(self.n_states, self.n_actions, self.args)
        self.transition = list()

        # actor-critic network setting
        self.actor = ActorSAC(self.n_states, self.n_actions, self.args)
        self.critic_eval = CriticTwin(self.n_states, self.n_actions, self.args)

        # loss function
        self.criterion = nn.MSELoss()

        # optimizer setting
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_eval.parameters(), lr=self.args.critic_lr)

        self.critic_target = copy.deepcopy(self.critic_eval)
        self.critic_target.eval()
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # Temperature Coefficient
        self.target_entropy = -self.n_actions
        self.log_alpha = T.zeros(1, requires_grad=True, device=self.args.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.args.alpha_lr)

        # Storage location creation
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)

        self.model_path = self.args.save_dir + '/' + args.algorithm
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        self.model_path = self.model_path + '/' + args.env_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        if os.path.exists(self.model_path + '/sac_actor.pth'):
            self.load_models()

        self.total_step = 0

        # Behavior Cloning parameters
        data_path = os.path.join('','bc_memo.npy')
        self.lambda1 = self.args.lambda1
        self.lambda2 = self.args.lambda2 / self.args.bc_batch_size
        self.memo = np.load(data_path, allow_pickle = True)

        self.bc_data = ReplayBuffer(self.n_states, self.n_actions, self.args, len(self.memo))
        self.bc_data.store_for_BC_data(self.memo)

    def choose_action(self, state, epsilon):
        with T.no_grad():
            if epsilon >= np.random.random() and not self.args.evaluate:
                choose_action = np.random.uniform(self.low_action, self.max_action, self.n_actions)
            else :
                choose_action, _ = self.actor(T.as_tensor(state, dtype=T.float32, device=self.actor.device))
                choose_action = choose_action.detach().cpu().numpy()
            self.transition = [state, choose_action]
        return choose_action

    def learn(self):
        q1_loss, q2_loss, state = self._value_update(self.memory, self.args.batch_size)
        critic_loss = q1_loss + q2_loss

        # update value
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # target network soft update
        if self.total_step % self.args.target_update_interval == 0:
            self._target_soft_update(self.critic_target, self.critic_eval, self.args.tau)

        for p in self.critic_eval.parameters():
            p.requires_grad = False

        pg_loss, new_log_prob = self._policy_update(state)

        alpha_loss = self._temperature_update(new_log_prob)

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        bc_loss = self._behavioral_cloning_update(self.bc_data, self.args.bc_batch_size)

        # Behavior Cloning with Actor Loss Function
        actor_loss = self.lambda1 * pg_loss + self.lambda2 * bc_loss

        # update policy
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for p in self.critic_eval.parameters():
            p.requires_grad = True

        self.alpha = self.log_alpha.exp()

    def save_models(self):
        print('------ Save model ------')
        _save_model(self.actor, self.actor_path)
        _save_model(self.critic_eval, self.critic_path)
        checkpoint = os.path.join(self.args.save_dir + '/' + self.args.algorithm +'/' + self.args.env_name, 'log_alpha.pth')
        T.save(self.log_alpha, checkpoint)

    def load_models(self):
        print('------ load model ------')
        _load_model(self.actor, self.actor_path)
        _load_model(self.critic_eval, self.critic_path)
        checkpoint = os.path.join(self.args.save_dir + '/' + self.args.algorithm +'/' + self.args.env_name, 'log_alpha.pth')
        self.log_alpha = T.load(checkpoint)

    # target network soft update
    def _target_soft_update(self, target_net, eval_net , tau=None):
        if tau == None:
            tau = self.args.tau
        with T.no_grad():
            for t_p, l_p in zip(target_net.parameters(), eval_net.parameters()):
                t_p.data.copy_(tau * l_p.data + (1 - tau) * t_p.data)

    def _value_update(self, buffer, batch_size):
        with T.no_grad():
            # Select data from ReplayBuffer with batch_size size
            samples = buffer.sample_batch(batch_size)

            state = T.as_tensor(samples['state'], dtype=T.float32, device=self.args.device)
            next_state = T.as_tensor(samples['next_state'], dtype=T.float32, device=self.args.device)
            action = T.as_tensor(samples['action'], dtype=T.float32, device=self.args.device).reshape(-1, 1)
            reward = T.as_tensor(samples['reward'], dtype=T.float32, device=self.args.device).reshape(-1,1)
            mask = T.as_tensor(samples['mask'], dtype=T.float32, device=self.args.device).reshape(-1,1)

            next_action, next_log_prob = self.actor(next_state)
            next_target_q1, next_target_q2 = self.critic_target.get_double_q(next_state, next_action)
            next_target_q = T.min(next_target_q1, next_target_q2)
            target_q = reward + (next_target_q - self.alpha * next_log_prob) * mask

        # Twin Critic Network Loss functions
        current_q1, current_q2 = self.critic_eval.get_double_q(state, action)
        # TD error
        q1_loss = self.criterion(current_q1, target_q)
        q2_loss = self.criterion(current_q2, target_q)
        return q1_loss, q2_loss, state

    def _policy_update(self, state):
        new_action, new_log_prob = self.actor(state)
        q_1, q_2 = self.critic_eval.get_double_q(state, new_action)
        q = T.min(q_1, q_2)
        actor_loss = (self.alpha * new_log_prob - q).mean()
        return actor_loss, new_log_prob

    def _temperature_update(self, new_log_prob):
        alpha_loss = -self.log_alpha * (new_log_prob.detach() + self.target_entropy).mean()
        return alpha_loss

    def _behavioral_cloning_update(self, buffer, batch_size):
        with T.no_grad():
            samples_bc = buffer.sample_batch(batch_size)

            state_bc = T.as_tensor(samples_bc['state'], dtype=T.float32, device=self.args.device)
            next_state_bc = T.as_tensor(samples_bc['next_state'], dtype=T.float32, device=self.args.device)
            action_bc = T.as_tensor(samples_bc['action'], dtype=T.float32, device=self.args.device).reshape(-1, 1)
            reward_bc = T.as_tensor(samples_bc['reward'], dtype=T.float32, device=self.args.device).reshape(-1,1)
            mask_bc = T.as_tensor(samples_bc['mask'], dtype=T.float32, device=self.args.device).reshape(-1,1)

        pred_action, _ = self.actor(state_bc)
        q_t = T.min(*self.critic_eval.get_double_q(state_bc, action_bc))
        q_e = T.min(*self.critic_eval.get_double_q(state_bc, pred_action))
        qf_mask = T.gt(q_t, q_e).to(self.args.device)
        qf_mask = qf_mask.float()
        n_qf_mask = int(qf_mask.sum().item())

        if n_qf_mask == 0:
            bc_loss = T.zeros(1, device=self.args.device)
        else:
            bc_loss = (
                T.mul(pred_action, qf_mask) - T.mul(action_bc, qf_mask)
            ).pow(2).sum() / n_qf_mask

        return bc_loss

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

# model save functions
def _save_model(net, dirpath):
    T.save(net.state_dict(), dirpath)

# model load functions
def _load_model(net, dirpath):
    net.load_state_dict(T.load(dirpath))