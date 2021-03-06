# Copyright (c) 2021: Zhiyuan Nan (namjw@hanyang.ac.kr).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import os
import copy
import gym
from gym.wrappers import RescaleAction

from test.network import QNetwork, DuelingNetwork, DuelingTwinNetwork
from test.network import Actor, ActorA2C, ActorPPO, ActorSAC
from test.network import CriticQ, CriticV, CriticTwin

from test.replaybuffer import ReplayBuffer, ReplayBufferPPO, PrioritizedReplayBuffer
from test.utils import OUNoise, _target_soft_update, _grad_false, compute_gae, ppo_iter
from test.utils import _save_model, _load_model

class DQNAgent:
    def __init__(self, args):

        self.args = args
        self.critic_update_function = None
        self.if_per = args.if_per

        # Environment setting
        self.env = gym.make(args.env_name)
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n

        self.checkpoint = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, args.algorithm_path)

        # network setting
        self.eval = QNetwork(self.n_states, self.n_actions, args)
        self.target = copy.deepcopy(self.eval)
        _grad_false(self.target)

        # optimizer setting
        self.optimizer = optim.Adam(self.eval.parameters(), lr=self.args.critic_lr)
        self.criterion = nn.MSELoss(reduction='none' if self.if_per else 'mean')

        # replay buffer
        if self.if_per:
            self.memory = PrioritizedReplayBuffer(self.n_states, self.args, self.args.buffer_size, self.args.alpha)
            self.critic_update_function = self._value_update_per
        else:
            self.memory = ReplayBuffer(self.n_states, self.n_actions, args)
            self.critic_update_function = self._value_update

        self.transition = list()

        self.total_step = 0
        self.learning_step = 0

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

    def learn(self, writer):
        if not self.memory.ready(self.args.batch_size):
            return
        self.learning_step += 1

        # update function
        critic_loss = self.critic_update_function(self.memory, self.args.batch_size)

        self.optimizer.zero_grad()
        critic_loss.backward()
        self.optimizer.step()

        if self.learning_step % 1000 == 0:
            writer.add_scalar("loss/critic", critic_loss.item(), self.learning_step)

        # target network hard update
        if self.learning_step % self.args.update_rate == 0:
            _target_soft_update(self.target, self.eval, 1.0)

    def save_models(self):
        print('------ Save model ------')
        _save_model(self.eval, self.checkpoint)

    def load_models(self):
        print('------ load model ------')
        _load_model(self.eval, self.checkpoint)

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

    def _value_update_per(self, buffer, batch_size):
        with T.no_grad():
            # Select data from ReplayBuffer with batch_size size
            samples = buffer.sample_batch(batch_size, self.args.beta)

            state = T.as_tensor(samples['state'], dtype=T.float32, device=self.args.device)
            next_state = T.as_tensor(samples['next_state'], dtype=T.float32, device=self.args.device)

            action = T.as_tensor(samples['action'], dtype=T.long, device=self.args.device).reshape(-1, 1)
            reward = T.as_tensor(samples['reward'], dtype=T.float32, device=self.args.device).reshape(-1, 1)
            mask = T.as_tensor(samples['mask'], dtype=T.float32, device=self.args.device).reshape(-1, 1)

            weights = T.as_tensor(samples["weights"], dtype=T.float32, device=self.args.device).reshape(-1, 1)
            indices = samples["indices"]

            next_q = self.target(next_state).max(dim=1, keepdim=True)[0]
            # Here we calculate action value Q(s,a) = R + yV(s')
            target_q = reward + next_q * mask

        current_q = self.eval(state).gather(1, action)
        elementwise_loss = self.criterion(current_q, target_q)

        loss = T.mean(elementwise_loss * weights)

        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.args.prior_eps
        buffer.update_priorities(indices, new_priorities)

        return loss

class DoubleDQNAgent:
    def __init__(self, args):

        self.args = args
        self.critic_update_function = None
        self.if_per = args.if_per

        # Environment setting
        self.env = gym.make(args.env_name)
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n

        self.checkpoint = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, args.algorithm_path)

        # network setting
        self.eval = QNetwork(self.n_states, self.n_actions, args)
        self.target = copy.deepcopy(self.eval)
        _grad_false(self.target)

        # optimizer setting
        self.optimizer = optim.Adam(self.eval.parameters(), lr=self.args.critic_lr)
        self.criterion = nn.MSELoss(reduction='none' if self.if_per else 'mean')

        # replay buffer
        if self.if_per:
            self.memory = PrioritizedReplayBuffer(self.n_states, self.args, self.args.buffer_size, self.args.alpha)
            self.critic_update_function = self._value_update_per
        else:
            self.memory = ReplayBuffer(self.n_states, self.n_actions, args)
            self.critic_update_function = self._value_update

        self.transition = list()

        self.total_step = 0
        self.learning_step = 0

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

    def learn(self, writer):
        if not self.memory.ready(self.args.batch_size):
            return
        self.learning_step += 1

        # TD error
        critic_loss = self.critic_update_function(self.memory, self.args.batch_size)

        self.optimizer.zero_grad()
        critic_loss.backward()
        self.optimizer.step()

        if self.learning_step % 1000 == 0:
            writer.add_scalar("loss/critic", critic_loss.item(), self.learning_step)

        # target network hard update
        if self.learning_step % self.args.update_rate == 0:
            _target_soft_update(self.target, self.eval, 1.0)

    def save_models(self):
        print('------ Save model ------')
        _save_model(self.eval, self.checkpoint)

    def load_models(self):
        print('------ load model ------')
        _load_model(self.eval, self.checkpoint)

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

    def _value_update_per(self, buffer, batch_size):
        with T.no_grad():
            # Select data from ReplayBuffer with batch_size size
            samples = buffer.sample_batch(batch_size, self.args.beta)

            state = T.as_tensor(samples['state'], dtype=T.float32, device=self.args.device)
            next_state = T.as_tensor(samples['next_state'], dtype=T.float32, device=self.args.device)

            action = T.as_tensor(samples['action'], dtype=T.long, device=self.args.device).reshape(-1, 1)
            reward = T.as_tensor(samples['reward'], dtype=T.float32, device=self.args.device).reshape(-1, 1)
            mask = T.as_tensor(samples['mask'], dtype=T.float32, device=self.args.device).reshape(-1, 1)

            weights = T.as_tensor(samples["weights"], dtype=T.float32, device=self.args.device).reshape(-1, 1)
            indices = samples["indices"]

            # Double DQN
            next_q = self.target(next_state).gather(1, self.eval(next_state).argmax(dim = 1, keepdim = True))
            # Here we calculate action value Q(s,a) = R + yV(s')
            target_q = reward + next_q * mask

        current_q = self.eval(state).gather(1, action)
        elementwise_loss = self.criterion(current_q, target_q)

        loss = T.mean(elementwise_loss * weights)

        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.args.prior_eps
        buffer.update_priorities(indices, new_priorities)

        return loss

class DuelingDQNAgent:
    def __init__(self, args):

        self.args = args
        self.critic_update_function = None
        self.if_per = args.if_per

        # Environment setting
        self.env = gym.make(args.env_name)
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n

        self.checkpoint = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, args.algorithm_path)

        # network setting
        self.eval = DuelingNetwork(self.n_states, self.n_actions, args)
        self.target = copy.deepcopy(self.eval)
        _grad_false(self.target)

        # optimizer setting
        self.optimizer = optim.Adam(self.eval.parameters(), lr=self.args.critic_lr)
        self.criterion = nn.MSELoss(reduction='none' if self.if_per else 'mean')

        # replay buffer
        if self.if_per:
            self.memory = PrioritizedReplayBuffer(self.n_states, self.args, self.args.buffer_size, self.args.alpha)
            self.critic_update_function = self._value_update_per
        else:
            self.memory = ReplayBuffer(self.n_states, self.n_actions, args)
            self.critic_update_function = self._value_update
        self.transition = list()

        self.total_step = 0
        self.learning_step = 0

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

    def learn(self, writer):
        if not self.memory.ready(self.args.batch_size):
            return
        self.learning_step += 1

        # update function
        critic_loss = self.critic_update_function(self.memory, self.args.batch_size)

        self.optimizer.zero_grad()
        critic_loss.backward()
        self.optimizer.step()

        if self.learning_step % 1000 == 0:
            writer.add_scalar("loss/critic", critic_loss.item(), self.learning_step)

        # target network hard update
        if self.learning_step % self.args.update_rate == 0:
            _target_soft_update(self.target, self.eval, 1.0)

    def save_models(self):
        print('------ Save model ------')
        _save_model(self.eval, self.checkpoint)

    def load_models(self):
        print('------ load model ------')
        _load_model(self.eval, self.checkpoint)

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

    def _value_update_per(self, buffer, batch_size):
        with T.no_grad():
            # Select data from ReplayBuffer with batch_size size
            samples = buffer.sample_batch(batch_size, self.args.beta)

            state = T.as_tensor(samples['state'], dtype=T.float32, device=self.args.device)
            next_state = T.as_tensor(samples['next_state'], dtype=T.float32, device=self.args.device)

            action = T.as_tensor(samples['action'], dtype=T.long, device=self.args.device).reshape(-1, 1)
            reward = T.as_tensor(samples['reward'], dtype=T.float32, device=self.args.device).reshape(-1, 1)
            mask = T.as_tensor(samples['mask'], dtype=T.float32, device=self.args.device).reshape(-1, 1)

            weights = T.as_tensor(samples["weights"], dtype=T.float32, device=self.args.device).reshape(-1, 1)
            indices = samples["indices"]

            next_q = self.target(next_state).max(dim=1, keepdim=True)[0]
            # Here we calculate action value Q(s,a) = R + yV(s')
            target_q = reward + next_q * mask

        current_q = self.eval(state).gather(1, action)
        elementwise_loss = self.criterion(current_q, target_q)

        loss = T.mean(elementwise_loss * weights)

        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.args.prior_eps
        buffer.update_priorities(indices, new_priorities)

        return loss

class D3QNAgent:
    def __init__(self, args):

        self.args = args
        self.critic_update_function = None
        self.if_per = args.if_per

        # Environment setting
        self.env = gym.make(args.env_name)
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n

        self.checkpoint = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, args.algorithm_path)
        # network setting
        self.eval = DuelingNetwork(self.n_states, self.n_actions, args)
        self.target = copy.deepcopy(self.eval)
        _grad_false(self.target)

        # optimizer setting
        self.optimizer = optim.Adam(self.eval.parameters(), lr=self.args.critic_lr)
        # loss function
        self.criterion = nn.MSELoss(reduction='none' if self.if_per else 'mean')

        # replay buffer
        if self.if_per:
            self.memory = PrioritizedReplayBuffer(self.n_states, self.args, self.args.buffer_size, self.args.alpha)
            self.critic_update_function = self._value_update_per
        else:
            self.memory = ReplayBuffer(self.n_states, self.n_actions, args)
            self.critic_update_function = self._value_update
        self.transition = list()

        self.total_step = 0
        self.learning_step = 0

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

    def learn(self, writer):
        if not self.memory.ready(self.args.batch_size):
            return
        self.learning_step += 1

        # TD error
        critic_loss = self.critic_update_function(self.memory, self.args.batch_size)

        self.optimizer.zero_grad()
        critic_loss.backward()
        self.optimizer.step()

        if self.learning_step % 1000 == 0:
            writer.add_scalar("loss/critic", critic_loss.item(), self.learning_step)

        # target network hard update
        if self.learning_step % self.args.update_rate == 0:
            _target_soft_update(self.target, self.eval, 1.0)

    def save_models(self):
        print('------ Save model ------')
        _save_model(self.eval, self.checkpoint)

    def load_models(self):
        print('------ load model ------')
        _load_model(self.eval, self.checkpoint)

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

    def _value_update_per(self, buffer, batch_size):
        with T.no_grad():
            # Select data from ReplayBuffer with batch_size size
            samples = buffer.sample_batch(batch_size, self.args.beta)

            state = T.as_tensor(samples['state'], dtype=T.float32, device=self.args.device)
            next_state = T.as_tensor(samples['next_state'], dtype=T.float32, device=self.args.device)

            action = T.as_tensor(samples['action'], dtype=T.long, device=self.args.device).reshape(-1, 1)
            reward = T.as_tensor(samples['reward'], dtype=T.float32, device=self.args.device).reshape(-1, 1)
            mask = T.as_tensor(samples['mask'], dtype=T.float32, device=self.args.device).reshape(-1, 1)

            weights = T.as_tensor(samples["weights"], dtype=T.float32, device=self.args.device).reshape(-1, 1)
            indices = samples["indices"]

            # Double DQN
            next_q = self.target(next_state).gather(1, self.eval(next_state).argmax(dim = 1, keepdim = True))
            # Here we calculate action value Q(s,a) = R + yV(s')
            target_q = reward + next_q * mask

        current_q = self.eval(state).gather(1, action)
        elementwise_loss = self.criterion(current_q, target_q)

        loss = T.mean(elementwise_loss * weights)

        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.args.prior_eps
        buffer.update_priorities(indices, new_priorities)

        return loss

class NoisyDQNAgent:
    def __init__(self, args):

        self.args = args
        self.critic_update_function = None
        self.if_per = args.if_per

        # Environment setting
        self.env = gym.make(args.env_name)
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n

        self.checkpoint = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, args.algorithm_path)

        # network setting
        self.eval = QNetwork(self.n_states, self.n_actions, args)
        self.target = copy.deepcopy(self.eval)
        _grad_false(self.target)

        # optimizer setting
        self.optimizer = optim.Adam(self.eval.parameters(), lr=self.args.critic_lr)

        # loss function
        self.criterion = nn.MSELoss(reduction='none' if self.if_per else 'mean')

        # replay buffer
        if self.if_per:
            self.memory = PrioritizedReplayBuffer(self.n_states, self.args, self.args.buffer_size, self.args.alpha)
            self.critic_update_function = self._value_update_per
        else:
            self.memory = ReplayBuffer(self.n_states, self.n_actions, args)
            self.critic_update_function = self._value_update
        self.transition = list()

        self.total_step = 0
        self.learning_step = 0

    def choose_action(self, state):
        with T.no_grad():
            choose_action = self.eval(T.as_tensor(state, dtype=T.float32, device=self.args.device)).argmax()
            choose_action = choose_action.detach().cpu().numpy()
            if not self.args.evaluate:
                self.transition = [state, choose_action]
        return choose_action

    def learn(self, writer):
        if not self.memory.ready(self.args.batch_size):
            return
        self.learning_step += 1

        # update function
        critic_loss = self.critic_update_function(self.memory, self.args.batch_size)

        self.optimizer.zero_grad()
        critic_loss.backward()
        self.optimizer.step()

        # noise update
        self.eval.reset_noise()
        self.target.reset_noise()

        if self.learning_step % 1000 == 0:
            writer.add_scalar("loss/critic", critic_loss.item(), self.learning_step)

        # target network hard update
        if self.learning_step % self.args.update_rate == 0:
            _target_soft_update(self.target, self.eval, 1.0)

    def save_models(self):
        print('------ Save model ------')
        _save_model(self.eval, self.checkpoint)

    def load_models(self):
        print('------ load model ------')
        _load_model(self.eval, self.checkpoint)

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

    def _value_update_per(self, buffer, batch_size):
        with T.no_grad():
            # Select data from ReplayBuffer with batch_size size
            samples = buffer.sample_batch(batch_size, self.args.beta)

            state = T.as_tensor(samples['state'], dtype=T.float32, device=self.args.device)
            next_state = T.as_tensor(samples['next_state'], dtype=T.float32, device=self.args.device)

            action = T.as_tensor(samples['action'], dtype=T.long, device=self.args.device).reshape(-1, 1)
            reward = T.as_tensor(samples['reward'], dtype=T.float32, device=self.args.device).reshape(-1, 1)
            mask = T.as_tensor(samples['mask'], dtype=T.float32, device=self.args.device).reshape(-1, 1)

            weights = T.as_tensor(samples["weights"], dtype=T.float32, device=self.args.device).reshape(-1, 1)
            indices = samples["indices"]

            next_q = self.target(next_state).max(dim=1, keepdim=True)[0]
            # Here we calculate action value Q(s,a) = R + yV(s')
            target_q = reward + next_q * mask

        current_q = self.eval(state).gather(1, action)
        elementwise_loss = self.criterion(current_q, target_q)

        loss = T.mean(elementwise_loss * weights)

        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.args.prior_eps
        buffer.update_priorities(indices, new_priorities)

        return loss

class DDPGAgent:
    def __init__(self, args):
        self.args = args

        self.critic_update_function = None
        self.if_per = args.if_per

        self.actor_path = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, args.file_actor)
        self.critic_path = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, args.file_critic)

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
        if self.if_per:
            self.memory = PrioritizedReplayBuffer(self.n_states, self.args, self.args.buffer_size, self.args.alpha)
            self.critic_update_function = self._value_update_per
        else:
            self.memory = ReplayBuffer(self.n_states, self.n_actions, args)
            self.critic_update_function = self._value_update
        self.transition = list()

        # actor-critic network setting
        self.actor_eval = Actor(self.n_states, self.n_actions, self.args)
        self.critic_eval = CriticQ(self.n_states, self.n_actions, self.args)

        self.actor_target = copy.deepcopy(self.actor_eval)
        _grad_false(self.actor_target)

        self.critic_target = copy.deepcopy(self.critic_eval)
        _grad_false(self.critic_target)

        # optimizer setting
        self.actor_optimizer = optim.Adam(self.actor_eval.parameters(), lr=self.args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_eval.parameters(), lr=self.args.critic_lr)

        # loss function
        self.criterion = nn.MSELoss(reduction='none' if self.if_per else 'mean')

        self.total_step = 0

        self.learning_step = 0

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

    def learn(self, writer):
        if not self.memory.ready(self.args.batch_size):
            return

        self.learning_step += 1

        # TD error
        critic_loss, state = self.critic_update_function(self.memory, self.args.batch_size)

        # update value
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.learning_step % 1000 == 0:
            writer.add_scalar("loss/critic", critic_loss.item(), self.learning_step)

        # critic target network soft update
        if self.total_step % self.args.target_update_interval == 0:
            _target_soft_update(self.critic_target, self.critic_eval, self.args.tau)

        for p in self.critic_eval.parameters():
            p.requires_grad = False

        # actor network loss function
        actor_loss = self._policy_update(state)

        # update policy
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.learning_step % 1000 == 0:
            writer.add_scalar("loss/actor", actor_loss.item(), self.learning_step)

        # actor target network soft update
        if self.total_step % self.args.target_update_interval == 0:
            _target_soft_update(self.actor_target, self.actor_eval, self.args.tau)

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

    def _value_update_per(self, buffer, batch_size):
        with T.no_grad():
            # Select data from ReplayBuffer with batch_size size
            samples = buffer.sample_batch(batch_size, self.args.beta)

            state = T.as_tensor(samples['state'], dtype=T.float32, device=self.args.device)
            next_state = T.as_tensor(samples['next_state'], dtype=T.float32, device=self.args.device)

            action = T.as_tensor(samples['action'], dtype=T.float32, device=self.args.device).reshape(-1, 1)
            reward = T.as_tensor(samples['reward'], dtype=T.float32, device=self.args.device).reshape(-1,1)
            mask = T.as_tensor(samples['mask'], dtype=T.float32, device=self.args.device).reshape(-1,1)

            weights = T.as_tensor(samples["weights"], dtype=T.float32, device=self.args.device).reshape(-1, 1)
            indices = samples["indices"]

            next_action = self.actor_target(next_state)
            next_value = self.critic_target(next_state, next_action)
            # Here we calculate action value Q(s,a) = R + yV(s')
            target_values = reward + next_value * mask

        eval_values = self.critic_eval(state, action)
        # TD error
        elementwise_loss = self.criterion(eval_values, target_values)
        critic_loss = T.mean(elementwise_loss * weights)

        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.args.prior_eps
        buffer.update_priorities(indices, new_priorities)

        return critic_loss, state

    def _policy_update(self, state):
        actor_loss = -self.critic_eval(state, self.actor_eval(state)).mean()
        return actor_loss

class TD3Agent:
    def __init__(self, args):
        self.args = args
        self.critic_update_function = None
        self.if_per = args.if_per
        self.actor_path = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, args.file_actor)
        self.critic_path = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, args.file_critic)

        # Environment setting
        self.env = gym.make(args.env_name)
        self.env = RescaleAction(self.env, -1, 1)
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]

        self.max_action = self.env.action_space.high[0]
        self.low_action = self.env.action_space.low[0]

        # replay buffer
        if self.if_per:
            self.memory = PrioritizedReplayBuffer(self.n_states, self.args, self.args.buffer_size, self.args.alpha)
            self.critic_update_function = self._value_update_per
        else:
            self.memory = ReplayBuffer(self.n_states, self.n_actions, args)
            self.critic_update_function = self._value_update
        self.transition = list()

        # actor-critic net setting
        self.actor_eval = Actor(self.n_states, self.n_actions, self.args)
        self.critic_eval = CriticTwin(self.n_states, self.n_actions, self.args)

        self.actor_target = copy.deepcopy(self.actor_eval)
        _grad_false(self.actor_target)

        self.critic_target = copy.deepcopy(self.critic_eval)
        _grad_false(self.critic_target)

        # optimizer setting
        self.actor_optimizer = optim.Adam(self.actor_eval.parameters(), lr=self.args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_eval.parameters(), lr=self.args.critic_lr)

        # loss function
        self.criterion = nn.MSELoss(reduction='none' if self.if_per else 'mean')

        self.total_step = 0

        self.learning_step = 0

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

    def learn(self, writer):
        if not self.memory.ready(self.args.batch_size):
            return

        self.learning_step += 1

        critic_loss, state = self._value_update(self.memory, self.args.batch_size)

        # update value
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.learning_step % 1000 == 0:
            writer.add_scalar("loss/critic", critic_loss.item(), self.learning_step)

        if self.total_step % self.args.policy_freq == 0:
            _target_soft_update(self.critic_target, self.critic_eval, self.args.tau)

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

            if self.learning_step % 1000 == 0:
                writer.add_scalar("loss/actor", actor_loss.item(), self.learning_step)

            # target network soft update
            _target_soft_update(self.actor_target, self.actor_eval, self.args.tau)

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

        critic_loss = q1_loss + q2_loss

        return critic_loss, state

    def _value_update_per(self, buffer, batch_size):
        with T.no_grad():
            # Select data from ReplayBuffer with batch_size size
            samples = buffer.sample_batch(batch_size, self.args.beta)

            state = T.as_tensor(samples['state'], dtype=T.float32, device=self.args.device)
            next_state = T.as_tensor(samples['next_state'], dtype=T.float32, device=self.args.device)
            action = T.as_tensor(samples['action'], dtype=T.float32, device=self.args.device).reshape(-1, 1)
            reward = T.as_tensor(samples['reward'], dtype=T.float32, device=self.args.device).reshape(-1,1)
            mask = T.as_tensor(samples['mask'], dtype=T.float32, device=self.args.device).reshape(-1,1)

            weights = T.as_tensor(samples["weights"], dtype=T.float32, device=self.args.device).reshape(-1, 1)
            indices = samples["indices"]

            noise = (T.randn_like(action) * self.args.policy_noise * self.max_action).clamp(self.args.noise_clip * self.low_action, self.args.noise_clip * self.max_action)
            next_action = (self.actor_target(next_state) + noise).clamp(self.low_action, self.max_action)

            next_target_q1, next_target_q2 = self.critic_target.get_double_q(next_state, next_action)
            next_target_q = T.min(next_target_q1, next_target_q2)
            # Here we calculate action value Q(s,a) = R + yV(s')
            target_q = reward + next_target_q * mask

        # Twin Critic Network Loss functions
        current_q1, current_q2 = self.critic_eval.get_double_q(state, action)

        # TD error
        elementwise_loss = self.criterion(T.min(current_q1, current_q2), target_q)
        critic_loss = T.mean((self.criterion(current_q1, target_q) + self.criterion(current_q2, target_q)) * weights)

        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.args.prior_eps
        buffer.update_priorities(indices, new_priorities)

        return critic_loss, state

    def _policy_update(self, state):
        actor_loss = -self.critic_eval(state, self.actor_eval(state)).mean()
        return actor_loss

class SACAgent:
    def __init__(self, args):
        self.args = args
        self.critic_update_function = None
        self.if_per = args.if_per
        self.actor_path = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, args.file_actor)
        self.critic_path = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, args.file_critic)

        # Environment setting
        self.env = gym.make(args.env_name)
        self.env = RescaleAction(self.env, -1, 1)
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]

        self.max_action = self.env.action_space.high[0]
        self.low_action = self.env.action_space.low[0]

        # replay buffer
        if self.if_per:
            self.memory = PrioritizedReplayBuffer(self.n_states, self.args, self.args.buffer_size, self.args.alpha)
            self.critic_update_function = self._value_update_per
        else:
            self.memory = ReplayBuffer(self.n_states, self.n_actions, args)
            self.critic_update_function = self._value_update
        self.transition = list()

        # actor-critic net setting
        self.actor = ActorSAC(self.n_states, self.n_actions, self.args)
        self.critic_eval = CriticTwin(self.n_states, self.n_actions, self.args)

        self.critic_target = copy.deepcopy(self.critic_eval)
        _grad_false(self.critic_target)

        # Temperature Coefficient
        self.target_entropy = -self.n_actions
        self.log_alpha = T.zeros(1, requires_grad=True, device=self.args.device)
        self.alpha = self.log_alpha.exp()

        # optimizer setting
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_eval.parameters(), lr=self.args.critic_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.args.alpha_lr)

        # loss function
        self.criterion = nn.MSELoss(reduction='none' if self.if_per else 'mean')

        self.total_step = 0

        self.learning_step = 0

    def choose_action(self, state, epsilon):
        with T.no_grad():
            if epsilon >= np.random.random() and not self.args.evaluate:
                choose_action = np.random.uniform(self.low_action, self.max_action, self.n_actions)
            else :
                choose_action, _ = self.actor(T.as_tensor(state, dtype=T.float32, device=self.actor.device))
                choose_action = choose_action.detach().cpu().numpy()
            self.transition = [state, choose_action]
        return choose_action

    def learn(self, writer):
        if not self.memory.ready(self.args.batch_size):
            return
        self.learning_step += 1

        # TD error
        # update value
        critic_loss, state = self._value_update(self.memory, self.args.batch_size)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.learning_step % 1000 == 0:
            writer.add_scalar("loss/critic", critic_loss.item(), self.learning_step)

        # target network soft update
        if self.total_step % self.args.target_update_interval == 0:
            _target_soft_update(self.critic_target, self.critic_eval, self.args.tau)

        for p in self.critic_eval.parameters():
            p.requires_grad = False

        actor_loss, new_log_prob = self._policy_update(state)

        # update Policy
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.learning_step % 1000 == 0:
            writer.add_scalar("loss/actor", actor_loss.item(), self.learning_step)

        # update Temperature Coefficient
        alpha_loss = self._temperature_update(new_log_prob)

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        if self.learning_step % 1000 == 0:
            writer.add_scalar("loss/alpha", alpha_loss.item(), self.learning_step)

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
        critic_loss = q1_loss + q2_loss
        return critic_loss, state

    def _value_update_per(self, buffer, batch_size):
        with T.no_grad():
            # Select data from ReplayBuffer with batch_size size
            samples = buffer.sample_batch(batch_size, self.args.beta)

            state = T.as_tensor(samples['state'], dtype=T.float32, device=self.args.device)
            next_state = T.as_tensor(samples['next_state'], dtype=T.float32, device=self.args.device)
            action = T.as_tensor(samples['action'], dtype=T.float32, device=self.args.device).reshape(-1, 1)
            reward = T.as_tensor(samples['reward'], dtype=T.float32, device=self.args.device).reshape(-1,1)
            mask = T.as_tensor(samples['mask'], dtype=T.float32, device=self.args.device).reshape(-1,1)

            weights = T.as_tensor(samples["weights"], dtype=T.float32, device=self.args.device).reshape(-1, 1)
            indices = samples["indices"]

            next_action, next_log_prob = self.actor(next_state)
            next_target_q1, next_target_q2 = self.critic_target.get_double_q(next_state, next_action)
            next_target_q = T.min(next_target_q1, next_target_q2)
            # Here we calculate action value Q(s,a) = R + yV(s')
            target_q = reward + (next_target_q - self.alpha * next_log_prob) * mask

        # Twin Critic Network Loss functions
        current_q1, current_q2 = self.critic_eval.get_double_q(state, action)

        # TD error
        elementwise_loss = self.criterion(T.min(current_q1, current_q2), target_q)
        critic_loss = T.mean((self.criterion(current_q1, target_q) + self.criterion(current_q2, target_q)) * weights)

        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.args.prior_eps
        buffer.update_priorities(indices, new_priorities)

        return critic_loss, state

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

class PPOAgent:
    def __init__(self, args):
        self.args = args
        self.actor_path = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, args.file_actor)
        self.critic_path = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, args.file_critic)
        # Environment setting
        self.env = gym.make(args.env_name)

        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]

        self.max_action = self.env.action_space.high[0]
        self.low_action = self.env.action_space.low[0]

        # actor-critic net setting
        self.actor = ActorPPO(self.n_states, self.n_actions, self.args)
        self.critic = CriticV(self.n_states, self.args)

        # optimizer setting
        self.optimizer = optim.Adam([{'params': self.actor.parameters(), 'lr': self.args.actor_lr},
                                    {'params': self.critic.parameters(), 'lr': self.args.critic_lr}])

        # loss function
        self.criterion = nn.MSELoss()

        self.total_step = 0
        self.learning_step = 0

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

    def learn(self, next_state, writer):
        self.learning_step += 1
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

            if self.learning_step % 1000 == 0:
                writer.add_scalar('Loss/actor', actor_loss.item(), self.learning_step)

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

            if self.learning_step % 1000 == 0:
                writer.add_scalar('Loss/critic', critic_loss.item(), self.learning_step)

            # PPO total loss function
            total_loss = self.args.value_weight * critic_loss + actor_loss - entropy * self.args.entropy_weight

            if self.learning_step % 1000 == 0:
                writer.add_scalar('Loss/total_loss', total_loss.item(), self.learning_step)

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

class A2CAgent:
    def __init__(self, args):
        self.args = args
        self.actor_path = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, args.file_actor)
        self.critic_path = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, args.file_critic)

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

        # optimizer setting
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.args.critic_lr)
        # Loss Function
        self.criterion = nn.SmoothL1Loss()

        self.total_step = 0
        self.learning_step = 0

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

    def learn(self, writer):
        self.learning_step += 1
        critic_loss, target_value, current_value, log_prob = self._value_update(self.transition)

        # update value
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.learning_step % 1000 == 0:
            writer.add_scalar('Loss/critic', critic_loss.item(), self.learning_step)

        actor_loss = self._policy_update(target_value, current_value, log_prob, self.args.entropy_weight)

        # update policy
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.learning_step % 1000 == 0:
            writer.add_scalar('Loss/actor', actor_loss.item(), self.learning_step)

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

class BC_SACAgent:
    def __init__(self, args):
        self.args = args
        self.actor_path = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, args.file_actor)
        self.critic_path = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, args.file_critic)

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

        self.critic_target = copy.deepcopy(self.critic_eval)
        _grad_false(self.critic_target)

        # loss function
        self.criterion = nn.MSELoss()

        # Temperature Coefficient
        self.target_entropy = -self.n_actions
        self.log_alpha = T.zeros(1, requires_grad=True, device=self.args.device)
        self.alpha = self.log_alpha.exp()

        # optimizer setting
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_eval.parameters(), lr=self.args.critic_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.args.alpha_lr)

        self.total_step = 0

        self.learning_step = 0

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

    def learn(self, writer):
        if not self.memory.ready(self.args.batch_size):
            return
        self.learning_step += 1

        q1_loss, q2_loss, state = self._value_update(self.memory, self.args.batch_size)
        critic_loss = q1_loss + q2_loss

        # update value
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.learning_step % 1000 == 0:
            writer.add_scalar("loss/critic", critic_loss.item(), self.learning_step)

        # target network soft update
        if self.total_step % self.args.target_update_interval == 0:
            _target_soft_update(self.critic_target, self.critic_eval, self.args.tau)

        for p in self.critic_eval.parameters():
            p.requires_grad = False

        pg_loss, new_log_prob = self._policy_update(state)

        alpha_loss = self._temperature_update(new_log_prob)

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        if self.learning_step % 1000 == 0:
            writer.add_scalar("loss/alpha", alpha_loss.item(), self.learning_step)

        bc_loss = self._behavioral_cloning_update(self.bc_data, self.args.bc_batch_size)

        # Behavior Cloning with Actor Loss Function
        actor_loss = self.lambda1 * pg_loss + self.lambda2 * bc_loss

        # update policy
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.learning_step % 1000 == 0:
            writer.add_scalar("loss/joint", actor_loss.item(), self.learning_step)
            writer.add_scalar("loss/actor", pg_loss.item(), self.learning_step)
            writer.add_scalar("loss/bc", bc_loss.item(), self.learning_step)

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
